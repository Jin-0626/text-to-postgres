"""Upload only the intended dataset package, never credentials or database dumps."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('repo_id',help='YOUR_USERNAME/postgresql-employees-assistant')
    parser.add_argument('--public',action='store_true',help='Create a public repository; default is private')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    api=HfApi()
    api.create_repo(repo_id=args.repo_id,repo_type='dataset',private=not args.public,exist_ok=True)
    api.upload_folder(repo_id=args.repo_id,repo_type='dataset',folder_path=str(root),
        allow_patterns=['README.md','HOW_TO.md','LICENSE.md','requirements.txt','*.py','schema_context.txt','validation_report.json','data/*.jsonl','metadata/*.jsonl'],
        commit_message='Upload schema-grounded PostgreSQL SFT starter dataset')
    print(f'https://huggingface.co/datasets/{args.repo_id}')

if __name__=='__main__': main()
