"""Validate contracts, split isolation and PostgreSQL syntax; optionally execute."""
import argparse
import json
import os
from pathlib import Path
from pglast import parse_sql
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true',help='Execute canonical SQL using DATABASE_URL and a restricted read-only role')
    args=parser.parse_args()
    load_dotenv(ROOT / '.env')
    conn=None
    if args.execute:
        import psycopg
        conn=psycopg.connect(os.environ['DATABASE_URL'],autocommit=True)
    seen=set(); families=set(); count=0; queries=0
    for split in ('train','validation','test'):
        rows=[json.loads(x) for x in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]
        metadata=[json.loads(x) for x in (ROOT/'metadata'/f'{split}.jsonl').read_text().splitlines()]
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
    report={'examples':count,'sql_queries':queries,'families':len(families),'json_contract':'passed','family_split_isolation':'passed','postgresql_parser':'passed','database_execution':'passed' if args.execute else 'not_run','semantic_result_correctness':'not_verified'}
    (ROOT/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
