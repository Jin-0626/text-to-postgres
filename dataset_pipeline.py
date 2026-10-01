"""Standalone synthetic PostgreSQL dataset pipeline: generate, validate, export, upload."""
import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
SQL_COLUMNS = ("question", "query", "schema")

SCHEMA = '''employees.employee(id bigint PK, birth_date date, first_name varchar(14), last_name varchar(16), gender enum M/F, hire_date date)
employees.department(id char(4) PK, dept_name varchar(40) UNIQUE)
employees.department_employee(employee_id bigint FK employee.id, department_id char(4) FK department.id, from_date date, to_date date; PK employee_id,department_id)
employees.department_manager(employee_id bigint FK employee.id, department_id char(4) FK department.id, from_date date, to_date date; PK employee_id,department_id)
employees.salary(employee_id bigint FK employee.id, amount bigint, from_date date, to_date date; PK employee_id,from_date)
employees.title(employee_id bigint FK employee.id, title varchar(50), from_date date, to_date nullable date; PK employee_id,title,from_date)'''
SYSTEM = '''You are a read-only PostgreSQL assistant. Return exactly one JSON object with keys action, sql, message. action is query, clarify, or unsupported. For query, sql contains one PostgreSQL SELECT/CTE statement and message is null. Otherwise sql is null and message is a concise question or explanation. Use only the supplied schema and schema-qualified table names. Never execute writes or invent columns or results. Active means from_date <= the specified date and the date < to_date; NULL title.to_date means open-ended. Do not infer currency, pay frequency, active date, or ambiguous business metrics. Historical record counts are not employee counts. Do not use CURRENT_DATE for unspecified dates. Queries are generated, not executed.'''

# Each tuple is a distinct query family. Parameter variants stay in one split.
FAMILIES = [
('employee_lookup','Look up employee {eid}: id, first name, last name and hire date.', 'SELECT id, first_name, last_name, hire_date FROM employees.employee WHERE id = {eid};'),
('hire_count','Count employees hired in calendar year {year}.', "SELECT COUNT(*) AS employee_count FROM employees.employee WHERE hire_date >= DATE '{year}-01-01' AND hire_date < DATE '{next_year}-01-01';"),
('hire_list','List the first {n} employees hired on or after {date}, ordered by hire date then id.', "SELECT id, first_name, last_name, hire_date FROM employees.employee WHERE hire_date >= DATE '{date}' ORDER BY hire_date, id LIMIT {n};"),
('name_search','Find employees whose last name is {name}, case-insensitively; order by id.', "SELECT id, first_name, last_name FROM employees.employee WHERE LOWER(last_name) = LOWER('{name}') ORDER BY id;"),
('birth_count','Count employees born in calendar year {year}.', "SELECT COUNT(*) AS employee_count FROM employees.employee WHERE birth_date >= DATE '{year}-01-01' AND birth_date < DATE '{next_year}-01-01';"),
('gender_counts','Count employees by recorded gender who were hired before {date}.', "SELECT gender, COUNT(*) AS employee_count FROM employees.employee WHERE hire_date < DATE '{date}' GROUP BY gender ORDER BY gender;"),
('annual_hires','Show hire counts by year for employees hired on or after {date}.', "SELECT EXTRACT(YEAR FROM hire_date)::integer AS hire_year, COUNT(*) AS employee_count FROM employees.employee WHERE hire_date >= DATE '{date}' GROUP BY 1 ORDER BY 1;"),
('departments','List departments whose names contain {fragment}, case-insensitively; order by id.', "SELECT id, dept_name FROM employees.department WHERE dept_name ILIKE '%{fragment}%' ORDER BY id;"),
('salary_history','Show all salary records for employee {eid}, earliest first.', 'SELECT employee_id, amount, from_date, to_date FROM employees.salary WHERE employee_id = {eid} ORDER BY from_date;'),
('salary_asof','Show all salary records active for employee {eid} on {date}.', "SELECT employee_id, amount, from_date, to_date FROM employees.salary WHERE employee_id = {eid} AND from_date <= DATE '{date}' AND DATE '{date}' < to_date ORDER BY from_date;"),
('salary_latest','Show the latest recorded salary row for employee {eid}, regardless of whether it is active.', 'SELECT employee_id, amount, from_date, to_date FROM employees.salary WHERE employee_id = {eid} ORDER BY from_date DESC LIMIT 1;'),
('salary_stats','Show minimum, maximum and average salary amount across salary records active on {date}. Count records, not people.', "SELECT MIN(amount) AS minimum_amount, MAX(amount) AS maximum_amount, AVG(amount) AS average_amount, COUNT(*) AS record_count FROM employees.salary WHERE from_date <= DATE '{date}' AND DATE '{date}' < to_date;"),
('salary_threshold','List salary records active on {date} with amount greater than {amount}; highest amount first, then employee id and from_date; limit {n}.', "SELECT employee_id, amount, from_date FROM employees.salary WHERE from_date <= DATE '{date}' AND DATE '{date}' < to_date AND amount > {amount} ORDER BY amount DESC, employee_id, from_date LIMIT {n};"),
('salary_missing','List the first {n} employees by id with no active salary row on {date}.', "SELECT e.id, e.first_name, e.last_name FROM employees.employee AS e WHERE NOT EXISTS (SELECT 1 FROM employees.salary AS s WHERE s.employee_id = e.id AND s.from_date <= DATE '{date}' AND DATE '{date}' < s.to_date) ORDER BY e.id LIMIT {n};"),
('salary_changes','For employee {eid}, show salary amount and its difference from the previous recorded amount, ordered by from_date.', 'SELECT employee_id, from_date, amount, amount - LAG(amount) OVER (PARTITION BY employee_id ORDER BY from_date) AS amount_change FROM employees.salary WHERE employee_id = {eid} ORDER BY from_date;'),
('salary_overlap','Find overlapping pairs of salary intervals for employee {eid}, using half-open intervals; order by the two start dates.', 'SELECT a.employee_id, a.from_date AS first_start, b.from_date AS second_start FROM employees.salary AS a JOIN employees.salary AS b ON a.employee_id = b.employee_id AND a.from_date < b.from_date AND a.from_date < b.to_date AND b.from_date < a.to_date WHERE a.employee_id = {eid} ORDER BY a.from_date, b.from_date;'),
('salary_bad_dates','List at most {n} salary records with from_date >= to_date, ordered by employee id and from_date.', 'SELECT employee_id, amount, from_date, to_date FROM employees.salary WHERE from_date >= to_date ORDER BY employee_id, from_date LIMIT {n};'),
('salary_record_count','Count historical salary rows for employee {eid}.', 'SELECT COUNT(*) AS salary_record_count FROM employees.salary WHERE employee_id = {eid};'),
('department_history','Show every department assignment for employee {eid}, with department names, ordered by from_date and department id.', 'SELECT de.department_id, d.dept_name, de.from_date, de.to_date FROM employees.department_employee AS de JOIN employees.department AS d ON d.id = de.department_id WHERE de.employee_id = {eid} ORDER BY de.from_date, de.department_id;'),
('department_asof','List distinct employees assigned to department {dept} on {date}, ordered by employee id; limit {n}.', "SELECT DISTINCT e.id, e.first_name, e.last_name FROM employees.employee AS e JOIN employees.department_employee AS de ON de.employee_id = e.id WHERE de.department_id = '{dept}' AND de.from_date <= DATE '{date}' AND DATE '{date}' < de.to_date ORDER BY e.id LIMIT {n};"),
('department_counts','Count distinct active employees per department on {date}; include zero-count departments and order by department id.', "SELECT d.id, d.dept_name, COUNT(DISTINCT de.employee_id) AS employee_count FROM employees.department AS d LEFT JOIN employees.department_employee AS de ON de.department_id = d.id AND de.from_date <= DATE '{date}' AND DATE '{date}' < de.to_date GROUP BY d.id, d.dept_name ORDER BY d.id;"),
('department_no_assignment','List the first {n} employees by id without a department assignment active on {date}.', "SELECT e.id FROM employees.employee AS e WHERE NOT EXISTS (SELECT 1 FROM employees.department_employee AS de WHERE de.employee_id = e.id AND de.from_date <= DATE '{date}' AND DATE '{date}' < de.to_date) ORDER BY e.id LIMIT {n};"),
('department_multi','Find employees assigned to more than one distinct department on {date}; order by employee id.', "SELECT employee_id, COUNT(DISTINCT department_id) AS department_count FROM employees.department_employee WHERE from_date <= DATE '{date}' AND DATE '{date}' < to_date GROUP BY employee_id HAVING COUNT(DISTINCT department_id) > 1 ORDER BY employee_id;"),
('department_ever','Count distinct employees who have ever been assigned to department {dept}.', "SELECT COUNT(DISTINCT employee_id) AS employee_count FROM employees.department_employee WHERE department_id = '{dept}';"),
('manager_asof','Show managers of department {dept} active on {date}, with employee names; order by employee id.', "SELECT e.id, e.first_name, e.last_name FROM employees.department_manager AS dm JOIN employees.employee AS e ON e.id = dm.employee_id WHERE dm.department_id = '{dept}' AND dm.from_date <= DATE '{date}' AND DATE '{date}' < dm.to_date ORDER BY e.id;"),
('manager_counts','Count distinct active managers per department on {date}, including zero counts; order by department id.', "SELECT d.id, d.dept_name, COUNT(DISTINCT dm.employee_id) AS manager_count FROM employees.department AS d LEFT JOIN employees.department_manager AS dm ON dm.department_id = d.id AND dm.from_date <= DATE '{date}' AND DATE '{date}' < dm.to_date GROUP BY d.id, d.dept_name ORDER BY d.id;"),
('manager_membership_gap','Find active managers on {date} without an active employee assignment in the same department; order by employee id and department id.', "SELECT dm.employee_id, dm.department_id FROM employees.department_manager AS dm WHERE dm.from_date <= DATE '{date}' AND DATE '{date}' < dm.to_date AND NOT EXISTS (SELECT 1 FROM employees.department_employee AS de WHERE de.employee_id = dm.employee_id AND de.department_id = dm.department_id AND de.from_date <= DATE '{date}' AND DATE '{date}' < de.to_date) ORDER BY dm.employee_id, dm.department_id;"),
('title_history','Show the title history of employee {eid}, ordered by from_date and title.', 'SELECT employee_id, title, from_date, to_date FROM employees.title WHERE employee_id = {eid} ORDER BY from_date, title;'),
('title_asof','Show all titles active for employee {eid} on {date}; treat NULL end dates as open-ended; order by title and start date.', "SELECT title, from_date, to_date FROM employees.title WHERE employee_id = {eid} AND from_date <= DATE '{date}' AND (to_date IS NULL OR DATE '{date}' < to_date) ORDER BY title, from_date;"),
('title_counts','Count distinct active employees per title on {date}, treating NULL end dates as open-ended; order by title.', "SELECT title, COUNT(DISTINCT employee_id) AS employee_count FROM employees.title WHERE from_date <= DATE '{date}' AND (to_date IS NULL OR DATE '{date}' < to_date) GROUP BY title ORDER BY title;"),
('title_search','List distinct employees whose active title on {date} is {title}; order by employee id; limit {n}.', "SELECT DISTINCT employee_id FROM employees.title WHERE title = '{title}' AND from_date <= DATE '{date}' AND (to_date IS NULL OR DATE '{date}' < to_date) ORDER BY employee_id LIMIT {n};"),
('title_null','List at most {n} title records with NULL end dates, ordered by employee id, title and start date.', 'SELECT employee_id, title, from_date FROM employees.title WHERE to_date IS NULL ORDER BY employee_id, title, from_date LIMIT {n};'),
('title_missing','List the first {n} employees by id without any active title on {date}; NULL end dates are open-ended.', "SELECT e.id FROM employees.employee AS e WHERE NOT EXISTS (SELECT 1 FROM employees.title AS t WHERE t.employee_id = e.id AND t.from_date <= DATE '{date}' AND (t.to_date IS NULL OR DATE '{date}' < t.to_date)) ORDER BY e.id LIMIT {n};"),
('salary_rank','Rank active salary records on {date} by amount descending using DENSE_RANK; show employee id, amount, rank; order by rank, employee id and start date; limit {n}.', "SELECT employee_id, amount, DENSE_RANK() OVER (ORDER BY amount DESC) AS salary_rank FROM employees.salary WHERE from_date <= DATE '{date}' AND DATE '{date}' < to_date ORDER BY salary_rank, employee_id, from_date LIMIT {n};"),
('salary_above_avg','List active salary records on {date} whose amount exceeds the average of active salary records on that date; order by amount descending, employee id and start date; limit {n}.', "WITH active AS (SELECT employee_id, amount, from_date FROM employees.salary WHERE from_date <= DATE '{date}' AND DATE '{date}' < to_date) SELECT employee_id, amount FROM active WHERE amount > (SELECT AVG(amount) FROM active) ORDER BY amount DESC, employee_id, from_date LIMIT {n};"),
('salary_percentile','Calculate the median amount across salary records active on {date}.', "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY amount) AS median_amount FROM employees.salary WHERE from_date <= DATE '{date}' AND DATE '{date}' < to_date;"),
('salary_department_avg','For department {dept} on {date}, average one active salary amount per distinct member. If active salary rows overlap, use the row with latest from_date. Exclude members without an active salary.', "WITH members AS (SELECT DISTINCT employee_id FROM employees.department_employee WHERE department_id = '{dept}' AND from_date <= DATE '{date}' AND DATE '{date}' < to_date), active_salary AS (SELECT DISTINCT ON (employee_id) employee_id, amount FROM employees.salary WHERE from_date <= DATE '{date}' AND DATE '{date}' < to_date ORDER BY employee_id, from_date DESC) SELECT AVG(s.amount) AS average_amount FROM members AS m JOIN active_salary AS s ON s.employee_id = m.employee_id;"),
('hire_after_birth','List at most {n} employees with hire_date before birth_date, ordered by id.', 'SELECT id, birth_date, hire_date FROM employees.employee WHERE hire_date < birth_date ORDER BY id LIMIT {n};'),
('salary_interval','Show salary records for employee {eid} overlapping [{date}, {end_date}), ordered by from_date.', "SELECT employee_id, amount, from_date, to_date FROM employees.salary WHERE employee_id = {eid} AND from_date < DATE '{end_date}' AND DATE '{date}' < to_date ORDER BY from_date;"),
('assignment_starts','Count department assignments starting in calendar year {year}, grouped by department id; order by department id.', "SELECT department_id, COUNT(*) AS assignment_count FROM employees.department_employee WHERE from_date >= DATE '{year}-01-01' AND from_date < DATE '{next_year}-01-01' GROUP BY department_id ORDER BY department_id;"),
]

BEHAVIORS = [
('missing_date','Show current salaries for employee {eid}.','clarify','Which as-of date should I use for active salaries?'),
('ambiguous_salary','Show the salary of employee {eid}.','clarify','Do you want the latest recorded salary, the full history, or salary active on a specified date?'),
('ambiguous_count','How many employees does department {dept} have?','clarify','Do you mean everyone ever assigned to the department, or active members on a specified date?'),
('missing_currency','Convert employee {eid} salary to MYR.','clarify','What is the source currency, which salary record should be used, and what exchange rate should apply?'),
('missing_email','Show the email address of employee {eid}.','unsupported','The supplied schema has no email column.'),
('missing_performance','Show performance ratings for employees in department {dept}.','unsupported','The supplied schema has no employee performance ratings.'),
('write_delete','Delete employee {eid} and all their records.','unsupported','This assistant is read-only; deletion requires a separate authorized write workflow.'),
('write_salary','Set employee {eid} salary to {amount}.','unsupported','This assistant is read-only; salary changes require a separate authorized write workflow.'),
('write_schema','Add an email column to employees.employee for employee {eid}.','unsupported','This assistant is read-only; schema changes require an authorized migration workflow.'),
('fabricated_answer','Without querying the database, tell me the exact salary of employee {eid}.','unsupported','I cannot know an employee salary without retrieving the relevant database records.'),
]

def generate_dataset(root: Path = ROOT):
    (root / 'data').mkdir(exist_ok=True)
    (root / 'metadata').mkdir(exist_ok=True)
    rng = random.Random(42)
    # Stratify SQL and behavioral families, never split a family's variants.
    assignments = {}
    for group in (FAMILIES, BEHAVIORS):
        ids = [x[0] for x in group]
        rng.shuffle(ids)
        a, b = int(len(ids)*.8), int(len(ids)*.9)
        assignments.update({f: 'train' if i<a else 'validation' if i<b else 'test' for i,f in enumerate(ids)})
    records = {s: [] for s in SPLITS}
    meta = {s: [] for s in records}
    for family in FAMILIES + BEHAVIORS:
        fid, request, target = family[:3]
        for i in range(8):
            year = 1993+i
            params = dict(eid=10001+i*137, year=year, next_year=year+1,
                          date=f'{year}-06-15', end_date=f'{year+1}-06-15',
                          n=5+i*5, amount=40000+i*5000, dept=f'd00{1+i%9}',
                          name=['Smith','Kumar','Lee','Wang','Garcia','Patel','Brown','Tan'][i],
                          fragment=['Sales','Research','Development','Human','Quality','Finance','Customer','Production'][i],
                          title=['Engineer','Senior Engineer','Staff','Senior Staff','Manager','Assistant Engineer','Technique Leader','Engineer'][i])
            action = 'query' if len(family)==3 else target
            output = dict(action=action, sql=target.format(**params) if action=='query' else None,
                          message=None if action=='query' else family[3])
            row = dict(prompt=[dict(role='system',content=SYSTEM),dict(role='user',content='Schema:\n'+SCHEMA+'\n\nRequest: '+request.format(**params))], completion=[dict(role='assistant',content=json.dumps(output,ensure_ascii=False))])
            split=assignments[fid]
            records[split].append(row)
            meta[split].append(dict(id=f'{fid}-{i:02}',family=fid,action=action,parameters=params,validation='pending_database_execution'))
    for split in records:
        for folder, rows in [('data',records[split]),('metadata',meta[split])]:
            (root/folder/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (root/'schema_context.txt').write_text(SCHEMA+'\n')
    print(json.dumps({s:len(rows) for s,rows in records.items()}))

def validate_dataset(root: Path = ROOT, execute: bool = False):
    from pglast import parse_sql
    from dotenv import load_dotenv
    load_dotenv(root / ".env")
    conn=None
    if execute:
        import psycopg
        conn=psycopg.connect(os.environ['DATABASE_URL'],autocommit=True)
    seen=set(); families=set(); count=0; queries=0
    for split in SPLITS:
        rows=[json.loads(x) for x in (root/'data'/f'{split}.jsonl').read_text().splitlines()]
        metadata=[json.loads(x) for x in (root/'metadata'/f'{split}.jsonl').read_text().splitlines()]
        assert len(rows)==len(metadata)
        current={x['family'] for x in metadata}
        assert not families & current, 'Family leakage'
        families |= current
        for row,meta in zip(rows,metadata):
            assert set(row)=={'prompt','completion'}
            assert [x['role'] for x in row['prompt']]==['system','user']
            assert len(row['completion'])==1 and row['completion'][0]['role']=='assistant'
            fingerprint=json.dumps(row,sort_keys=True)
            assert fingerprint not in seen, 'Duplicate example'
            seen.add(fingerprint)
            output=json.loads(row['completion'][0]['content'])
            assert set(output)=={'action','sql','message'}
            assert output['action']==meta['action']
            if output['action']=='query':
                assert output['message'] is None
                statements=parse_sql(output['sql'])
                assert len(statements)==1
                assert type(statements[0].stmt).__name__=='SelectStmt'
                queries+=1
                if conn:
                    with conn.transaction():
                        conn.execute('SET TRANSACTION READ ONLY')
                        conn.execute("SET LOCAL statement_timeout = '10s'")
                        cursor=conn.execute(output['sql'])
                        cursor.fetchall()
            else:
                assert output['action'] in ('clarify','unsupported')
                assert output['sql'] is None and output['message']
            count+=1
    if conn: conn.close()
    report={'examples':count,'sql_queries':queries,'families':len(families),'json_contract':'passed','family_split_isolation':'passed','postgresql_parser':'passed','database_execution':'passed' if execute else 'not_run','semantic_result_correctness':'not_verified'}
    (root/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

    return report

def convert_dataset(root: Path = ROOT):
    output_dir = root / "converted"
    output_dir.mkdir(exist_ok=True)

    behaviors = []

    for split in SPLITS:
        converted = []

        source = root / "data" / f"{split}.jsonl"
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue

            row = json.loads(line)
            user_text = next(
                message["content"]
                for message in row["prompt"]
                if message["role"] == "user"
            )

            schema, separator, question = user_text.partition("\n\nRequest:")
            if not separator or not schema.startswith("Schema:\n"):
                raise ValueError(f"{split}, line {line_number}: unexpected format")

            schema = schema.removeprefix("Schema:\n").strip()
            answer = json.loads(row["completion"][0]["content"])

            if answer["action"] != "query":
                behaviors.append({
                    "split": split,
                    "question": question.strip(),
                    "schema": schema,
                    "action": answer["action"],
                    "message": answer["message"],
                })
                continue

            converted.append({
                "question": question.strip(),
                "query": answer["sql"],
                "schema": schema,
            })

        target = output_dir / f"{split}.jsonl"
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
            encoding="utf-8",
        )
        print(f"{split}: {len(converted)} SQL examples")

    (root / "behavior_examples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in behaviors),
        encoding="utf-8",
    )
    print(f"Separate behavioral examples: {len(behaviors)}")

def load_converted_dataset(root: Path, splits=SPLITS):
    from datasets import load_dataset

    return load_dataset(
        "json",
        data_files={split: str(root / "converted" / f"{split}.jsonl") for split in splits},
    )


def assert_sql_columns(rows):
    assert set(rows.column_names) == set(SQL_COLUMNS)

def export_parquet(root: Path, loader=None):
    output = root / "parquet"
    output.mkdir(exist_ok=True)
    dataset = (loader or load_converted_dataset)(root)
    for split, rows in dataset.items():
        assert_sql_columns(rows)
        rows.to_parquet(str(output / f"{split}.parquet"))
        print(f"{split}: {len(rows)} rows exported")
    return dataset



def upload_parquet(root: Path, repo_id: str, public: bool = False):
    """Upload only a dataset card and the three Parquet splits in one commit."""
    from huggingface_hub import HfApi, CommitOperationAdd

    files = {f"data/{split}.parquet": root / "parquet" / f"{split}.parquet" for split in SPLITS}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    card = """---
language:
- en
license: cc-by-sa-3.0
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
---
# PostgreSQL Employees Text-to-SQL

Synthetic schema-grounded SQL examples with columns `question`, `query`, and `schema`.
The splits contain 256 training, 32 validation, and 32 test queries. Parameter variants
of a query family remain in one split. Clarification and unsupported examples are
retained locally and excluded from these Parquet files.

PostgreSQL syntax is validated; semantic result correctness is not verified.
Source schema: https://github.com/h8/employees-database. Original credits:
Fusheng Wang, Carlo Zaniolo, Giuseppe Maxia, and Patrick Crews; PostgreSQL
conversion maintained by h8. Changes: synthetic requests, queries, and family splits.
Distribution: https://creativecommons.org/licenses/by-sa/3.0/
"""
    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for remote, local in files.items()]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8")))
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=not public, exist_ok=True)
    api.create_commit(repo_id=repo_id, repo_type="dataset", operations=operations,
                      commit_message="Upload validated PostgreSQL text-to-SQL Parquet splits")
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(url)
    return url


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", nargs="?", help="YOUR_USERNAME/dataset-name; omit for a local-only run")
    parser.add_argument("--public", action="store_true", help="Create a public repository; new repositories default to private")
    parser.add_argument("--execute", action="store_true", help="Validate SQL execution using DATABASE_URL")
    parser.add_argument("--output-dir", type=Path, default=ROOT, help="Root directory for generated artifacts")
    args = parser.parse_args(argv)
    if args.public and not args.repo_id:
        parser.error("--public requires repo_id")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    generate_dataset(root)
    validate_dataset(root, execute=args.execute)
    convert_dataset(root)
    export_parquet(root)
    if args.repo_id:
        upload_parquet(root, args.repo_id, public=args.public)


if __name__ == "__main__":
    main()
