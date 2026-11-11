"""확정 자료의 해시를 확인한다.

    python tools/verify_final.py            # reports/final/sha256.json과 비교
    python tools/verify_final.py --update   # 확정 자료를 바꾼 뒤 해시를 다시 기록 (PR에 이유를 적을 것)

대상: data/final/, data/regime/, data/external/, config/*.yaml, reports/final/ (README·sha256.json 제외)
"""
import hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/final/sha256.json'
TARGETS = ['data/final', 'data/regime', 'data/external', 'config', 'reports/final']
SKIP = {'reports/final/sha256.json', 'reports/final/README.md'}


def files():
    for t in TARGETS:
        p = ROOT / t
        for f in sorted(p.rglob('*') if p.is_dir() else [p]):
            rel = f.relative_to(ROOT).as_posix()
            if f.is_file() and rel not in SKIP and (t != 'config' or f.suffix == '.yaml'):
                yield rel, f


def sha(f):
    h = hashlib.sha256()
    with open(f, 'rb') as r:
        for b in iter(lambda: r.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    now = {rel: sha(f) for rel, f in files()}
    if '--update' in sys.argv:
        OUT.write_text(json.dumps(now, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
        print(f'기록: {len(now)}개 → {OUT.relative_to(ROOT)}')
        return
    saved = json.loads(OUT.read_text(encoding='utf-8'))
    bad = [k for k in saved if now.get(k) != saved[k]]
    new = [k for k in now if k not in saved]
    for k in bad:
        print('다름 또는 없음:', k)
    for k in new:
        print('기록에 없음:', k)
    if bad or new:
        sys.exit(f'검증 실패: {len(bad)}개 다름, {len(new)}개 미기록. 의도한 변경이면 --update 후 PR에 이유를 적는다.')
    print(f'검증 통과: {len(saved)}개 파일')


if __name__ == '__main__':
    main()
