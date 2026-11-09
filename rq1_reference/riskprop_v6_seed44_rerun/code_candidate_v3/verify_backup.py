import argparse
from backup_seed44 import verify_backup
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True);a=parser.parse_args()
    m=verify_backup(a.archive)
    print('CRC and SHA256 PASS; files:',len(m['files']),'committed:',m['committed_artifacts_verified'])
