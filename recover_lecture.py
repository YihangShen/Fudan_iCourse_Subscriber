"""Repair only the requested lecture using cached transcript and PPT text."""
import base64
import json
from pathlib import Path
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from src.data.database import Database
from src.ai.summarizer import Summarizer
from src.ai.bucketer import assemble


def main():
    db=Database('data/icourse.db')
    output=[]
    try:
        rows=db.conn.execute('''SELECT l.*, c.title AS course_title, c.teacher FROM lectures l
            JOIN courses c USING(course_id) WHERE l.course_id=? AND (l.date=? OR l.sub_title LIKE ?)''',
            ('37535','2026-10-12','2026-10-12%')).fetchall()
        if not rows:
            raise RuntimeError('Requested lecture was not found')
        for row in rows:
            item=dict(row)
            if not (item['summary'] or '').strip():
                if not (item['transcript'] or '').strip():
                    raise RuntimeError('Cached transcript missing; automatic repair stopped')
                content, mode=assemble(item['transcript'],None,db.get_done_ppt_pages(item['sub_id']))
                summary, model=Summarizer().summarize(item['course_title'],content)
                db.update_summary(item['sub_id'],summary,model)
                db.mark_processed(item['sub_id'])
                db.clear_error(item['sub_id'])
                item['summary']=summary
                item['summary_model']=model
                print(f'Regenerated complete notes: {len(summary)} characters; queued for next delivery.')
            output.append({k:item[k] for k in ['sub_id','course_id','sub_title','date','course_title','teacher','summary','summary_model']})
    finally:
        db.conn.close()
    raw=json.dumps(output,ensure_ascii=False).encode('utf-8')
    key=get_random_bytes(32)
    cipher=AES.new(key,AES.MODE_GCM)
    ciphertext,tag=cipher.encrypt_and_digest(raw)
    public=RSA.import_key(Path('export-public.pem').read_bytes())
    wrapped=PKCS1_OAEP.new(public,hashAlgo=SHA256).encrypt(key)
    envelope={k:base64.b64encode(v).decode() for k,v in {'key':wrapped,'nonce':cipher.nonce,'tag':tag,'ciphertext':ciphertext}.items()}
    Path('recovered-lecture.enc.json').write_text(json.dumps(envelope))

if __name__=='__main__':
    main()
