from __future__ import annotations
import argparse, hashlib, json, random, re
from pathlib import Path
import fitz, joblib, numpy as np, pandas as pd, pytesseract, torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATA=ROOT/'resumee'/'Resumes PDF'
OUT=ROOT/'results'/'resume_training'
MODEL_OUT=ROOT/'models'/'distilbert_resume_classifier_v2'
BASE_MODEL='distilbert-base-uncased'
TESSERACT_EXE=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ALIASES={'accountant resumes':'Accountant','data science resumes':'Data Science','datascience':'Data Science','business analyst resumes':'Business Analyst','businessanalyst':'Business Analyst','devops engineer resumes':'DevOps Engineer','devopsengineer':'DevOps Engineer','python developer resumes':'Python Developer','pythondeveloper':'Python Developer','react':'React Developer','react developer resumes':'React Developer'}
AMBIGUOUS={'consult','consultant','nse','dot','pbo','pmo'}

def clean(s):
    s=(s or '').replace('\x00',' '); s=re.sub(r'[ \t]+',' ',s); return re.sub(r'\n{3,}','\n\n',s).strip()
def label_for(folder):
    key=re.sub(r'\s+',' ',folder.strip()).casefold()
    if key in ALIASES:return ALIASES[key],False,'obvious category-name variant'
    if key in AMBIGUOUS:return folder.strip(),True,'ambiguous category name; verify manually'
    return folder.strip(),False,''
def extract_pdf(path,dpi=180):
    try: doc=fitz.open(path)
    except Exception as e:return '','failed',f'open error: {e}'
    try:
        embedded=clean('\n'.join(p.get_text('text') or '' for p in doc))
        if len(embedded)>=80: doc.close(); return embedded,'embedded_text',''
        if Path(TESSERACT_EXE).exists():pytesseract.pytesseract.tesseract_cmd=TESSERACT_EXE
        parts=[]
        for page in doc:
            pix=page.get_pixmap(dpi=dpi,alpha=False); im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
            parts.append(pytesseract.image_to_string(im,lang='eng'))
        doc.close(); text=clean('\n'.join(parts))
        return (text,'tesseract_ocr','') if len(text)>=80 else (text,'ocr_short','less than 80 characters extracted')
    except Exception as e:
        try:doc.close()
        except Exception:pass
        return '','failed',f'OCR error: {e}'
def prepare(dataset,limit=0,resume=False,dpi=180):
    OUT.mkdir(parents=True,exist_ok=True); checkpoint=OUT/'raw_extraction_checkpoint.csv'
    old=pd.read_csv(checkpoint,keep_default_na=False) if resume and checkpoint.exists() else pd.DataFrame()
    done=set(old.source_path.astype(str)) if not old.empty else set()
    pdfs=sorted((p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower()=='.pdf'),key=lambda p:str(p).casefold())
    pdfs=[p for p in pdfs if str(p.resolve()) not in done]
    if limit>0:pdfs=pdfs[:limit]
    rows=old.to_dict('records') if not old.empty else []
    print(f'PDFs found: {len(list(dataset.rglob("*.pdf")))}; processing this run: {len(pdfs)}')
    print('OCR runs locally. Source PDFs are not modified.')
    for i,p in enumerate(pdfs,1):
        rel=p.relative_to(dataset); folder=rel.parts[0] if len(rel.parts)>1 else 'Uncategorized'; label,review,reason=label_for(folder)
        try:file_hash=hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:file_hash=''
        text,method,error=extract_pdf(p,dpi); th=hashlib.sha256(text.encode('utf-8',errors='ignore')).hexdigest() if text else ''
        rows.append({'source_path':str(p.resolve()),'relative_path':str(rel),'source_category':folder,'label':label,'label_needs_review':review,'label_review_reason':reason,'file_sha256':file_hash,'text_sha256':th,'text':text,'text_chars':len(text),'extraction_method':method,'error':error})
        if i%25==0:
            pd.DataFrame(rows).to_csv(checkpoint,index=False,encoding='utf-8-sig'); print(f'Checkpoint saved: {len(rows)} records')
    raw=pd.DataFrame(rows); raw.to_csv(checkpoint,index=False,encoding='utf-8-sig')
    if raw.empty:print('No PDFs processed. Check dataset path.'); return
    raw[raw.error.astype(str)!=''][['source_path','relative_path','source_category','extraction_method','text_chars','error']].to_csv(OUT/'ocr_errors.csv',index=False,encoding='utf-8-sig')
    review=raw.label_needs_review.astype(str).str.lower().isin(['true','1']) | (raw.text_chars.astype(int)<80)
    raw.loc[review,['source_path','relative_path','source_category','label','label_needs_review','label_review_reason','text_chars','extraction_method','error']].to_csv(OUT/'label_review.csv',index=False,encoding='utf-8-sig')
    usable=raw[raw.text.fillna('').astype(str).str.len()>=80].copy()
    usable['text_sha256']=usable.text.astype(str).map(lambda s:hashlib.sha256(s.encode('utf-8',errors='ignore')).hexdigest())
    dup=usable.duplicated('text_sha256',keep='first')
    usable.loc[dup,['source_path','relative_path','source_category','label','text_sha256']].to_csv(OUT/'duplicate_texts.csv',index=False,encoding='utf-8-sig')
    usable=usable.loc[~dup].copy(); usable=usable[~usable.label_needs_review.astype(str).str.lower().isin(['true','1'])]
    usable[['source_path','relative_path','source_category','label','text','text_chars','extraction_method','text_sha256']].to_csv(OUT/'resume_dataset.csv',index=False,encoding='utf-8-sig')
    usable.groupby('label').size().reset_index(name='resume_count').sort_values('resume_count',ascending=False).to_csv(OUT/'category_counts.csv',index=False,encoding='utf-8-sig')
    print('Prepared:',OUT/'resume_dataset.csv'); print('Review category_counts.csv and label_review.csv before training.')

class ResumeDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length):
        self.items=tokenizer(list(texts),truncation=True,padding='max_length',max_length=max_length); self.labels=list(map(int,labels))
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):
        item={k:torch.tensor(v[i],dtype=torch.long) for k,v in self.items.items()}; item['labels']=torch.tensor(self.labels[i],dtype=torch.long); return item

def split_data(df,seed=42):
    rng=random.Random(seed); tr=[]; va=[]; te=[]
    for _,g in df.groupby('label'):
        ids=list(g.index); rng.shuffle(ids); n=len(ids)
        if n<3:continue
        nt=max(1,round(n*.15)); nv=max(1,round(n*.15))
        while nt+nv>=n:
            if nv>1:nv-=1
            elif nt>1:nt-=1
            else:break
        te+=ids[:nt]; va+=ids[nt:nt+nv]; tr+=ids[nt+nv:]
    return df.loc[tr].copy(),df.loc[va].copy(),df.loc[te].copy()
def train(data_path,epochs=3,max_length=384,batch_size=8,min_class_count=5,seed=42):
    set_seed(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    df=pd.read_csv(data_path,keep_default_na=False); df['text']=df.text.astype(str).str.strip(); df['label']=df.label.astype(str).str.strip()
    df=df[(df.text.str.len()>=80)&(df.label!='')].copy(); counts=df.label.value_counts(); eligible=counts[counts>=min_class_count].index
    df[~df.label.isin(eligible)].to_csv(OUT/'model_v2_excluded_rare_classes.csv',index=False,encoding='utf-8-sig'); df=df[df.label.isin(eligible)].copy()
    if df.label.nunique()<2:raise ValueError('Need at least two career categories with enough samples.')
    df['hash_check']=df.text.map(lambda s:hashlib.sha256(s.encode('utf-8',errors='ignore')).hexdigest())
    if df.hash_check.duplicated().any():raise ValueError('Duplicate text found; re-run preparation and inspect duplicate_texts.csv.')
    tr,va,te=split_data(df,seed); common=set(tr.label)&set(va.label)&set(te.label)
    tr=tr[tr.label.isin(common)].reset_index(drop=True); va=va[va.label.isin(common)].reset_index(drop=True); te=te[te.label.isin(common)].reset_index(drop=True); labels=sorted(common)
    if len(labels)<2:raise ValueError('Too few categories appear in all splits; inspect category_counts.csv or lower --min-class-count.')
    enc=LabelEncoder(); enc.fit(labels); ytr=enc.transform(tr.label); yv=enc.transform(va.label); yte=enc.transform(te.label)
    tok=AutoTokenizer.from_pretrained(BASE_MODEL); model=AutoModelForSequenceClassification.from_pretrained(BASE_MODEL,num_labels=len(labels),id2label={i:s for i,s in enumerate(enc.classes_)},label2id={s:i for i,s in enumerate(enc.classes_)})
    dtr=ResumeDataset(tr.text,ytr,tok,max_length); dva=ResumeDataset(va.text,yv,tok,max_length); dte=ResumeDataset(te.text,yte,tok,max_length)
    def metrics(x):
        logits,gold=x; pred=np.argmax(logits,axis=1); return {'accuracy':accuracy_score(gold,pred),'macro_f1':f1_score(gold,pred,average='macro',zero_division=0)}
    args=TrainingArguments(output_dir=str(OUT/'model_v2_checkpoints'),num_train_epochs=epochs,learning_rate=2e-5,per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,weight_decay=.01,eval_strategy='epoch',save_strategy='epoch',logging_steps=50,load_best_model_at_end=True,metric_for_best_model='macro_f1',greater_is_better=True,save_total_limit=2,report_to='none',fp16=torch.cuda.is_available(),dataloader_num_workers=0,seed=seed)
    trainer=Trainer(model=model,args=args,train_dataset=dtr,eval_dataset=dva,processing_class=tok,compute_metrics=metrics)
    print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print(f'Classes: {len(labels)} | Train/validation/test: {len(tr)}/{len(va)}/{len(te)}')
    trainer.train(); val_metrics=trainer.evaluate(); out=trainer.predict(dte); pred=np.argmax(out.predictions,axis=1); gold=out.label_ids
    report=classification_report(gold,pred,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0)
    report_txt=classification_report(gold,pred,labels=list(range(len(labels))),target_names=labels,zero_division=0)
    result={'accuracy':float(accuracy_score(gold,pred)),'macro_f1':float(f1_score(gold,pred,average='macro',zero_division=0)),'weighted_f1':float(f1_score(gold,pred,average='weighted',zero_division=0)),'test_samples':len(te),'classes':len(labels),'epochs':epochs,'max_length':max_length,'batch_size':batch_size,'validation_metrics':{k:float(v) for k,v in val_metrics.items() if isinstance(v,(int,float,np.number))}}
    MODEL_OUT.mkdir(parents=True,exist_ok=True); (OUT/'model_v2').mkdir(parents=True,exist_ok=True); trainer.save_model(str(MODEL_OUT)); tok.save_pretrained(str(MODEL_OUT)); joblib.dump(enc,MODEL_OUT/'label_encoder.pkl')
    (OUT/'model_v2'/'test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); (OUT/'model_v2'/'classification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); (OUT/'model_v2'/'classification_report.txt').write_text(report_txt,encoding='utf-8')
    pd.DataFrame(confusion_matrix(gold,pred,labels=list(range(len(labels)))),index=labels,columns=labels).to_csv(OUT/'model_v2'/'confusion_matrix.csv',encoding='utf-8-sig')
    print('\nTEST METRICS\n',json.dumps(result,indent=2),'\n\n',report_txt); print('New model saved:',MODEL_OUT,'(old model untouched)')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,default=DEFAULT_DATA); p.add_argument('--prepare-only',action='store_true'); p.add_argument('--limit',type=int,default=0); p.add_argument('--resume',action='store_true'); p.add_argument('--ocr-dpi',type=int,default=180); p.add_argument('--data',type=Path,default=OUT/'resume_dataset.csv'); p.add_argument('--epochs',type=int,default=3); p.add_argument('--max-length',type=int,default=384); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--min-class-count',type=int,default=5); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    if not a.dataset.exists():raise FileNotFoundError(f'Dataset folder not found: {a.dataset}')
    prepare(a.dataset,a.limit,a.resume,a.ocr_dpi)
    if a.prepare_only:print('Preparation-only mode. Review output CSVs before training.'); return
    if not a.data.exists():raise FileNotFoundError(f'Prepared CSV not found: {a.data}')
    train(a.data,a.epochs,a.max_length,a.batch_size,a.min_class_count,a.seed)
if __name__=='__main__':main()
