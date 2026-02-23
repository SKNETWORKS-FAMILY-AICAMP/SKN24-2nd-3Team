import joblib
import json

from datetime import datetime
from pathlib import Path


class ModelManager:
  """
  머신러닝 모델 저장, 로드, 메타데이터 관리를 위한 클래스
  """
  
  def __init__(self, base_dir = 'models'):
    self.base_dir = Path(base_dir)
    self.base_dir.mkdir(parents=True, exist_ok=True)
    
  
  def save(self, model, model_name, metadata=None, compress = 3):
    """
    모델 파일 저장
    메타데이터는 json 파일로 함께 저장

    Args:
      model: 저장 모델 객체
      model_name: 모델명(확장자 제외)
      metadata: 모델 관련 메타데이터 (metrics, params)
      compress: joblib 압축 레벨 (default 3)

    Returns:
      model_path: 저장된 모델 파일 경로
    """
    model_path = self.base_dir / f'{model_name}.joblib'
    joblib.dump(model, model_path, compress=compress)
    
    meta = self._build_metadata(model, model_name, metadata)
    meta_path = self.base_dir / f'{model_name}_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
      json.dump(meta, f, ensure_ascii=False, indent=4)
    
    size = self._fmt_size(model_path.stat().st_size)
    
    print(f'[Model Manager]: {model_name} saved successfully ({size})')
    return str(model_path.resolve())
  
  
  def load(self, model_name):
    """
    모델과 메타데이터 로드
    
    Returns: 
      (model, metadata): 튜플
    """
    model_path = self.base_dir / f'{model_name}.joblib'
    if not model_path.exists():
      raise FileNotFoundError(f'[Model Manager]: Model file not found: {model_path}')
    
    model = joblib.load(model_path)
    
    meta_path = self.base_dir / f'{model_name}_meta.json'
    metadata = None
    if meta_path.exists():
      with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    print(f'[Model Manager]: {model_name} loaded successfully')
    return model, metadata
  
  
  def list_models(self):
    """
    저장된 모든 모델과 메타데이터 목록 반환
    """
    models = []
    for meta_file in sorted(self.base_dir.glob('*_meta.json')):
      with open(meta_file, 'r', encoding='utf-8') as f:
        models.append(json.load(f))
        
    return models
  
  
  def delete(self, model_name):
    """
    모델과 메타데이터 삭제
    """
    deleted = False
    for ext in ('.joblib', '_meta.json'):
      file_path = self.base_dir / f'{model_name}{ext}'
      if file_path.exists():
        file_path.unlink()
        deleted = True
    
    if deleted:
      print(f'[Model Manager]: {model_name} deleted successfully')
    else:
      print(f'[Model Manager]: No files found for {model_name}')
  
  
  def _build_metadata(self, model, model_name, extra=None):
    """
    모델 메타데이터 생성
    """
    meta = {
      'model_name': model_name,
      'model_type': type(model).__name__,
      'created_at': datetime.now().isoformat()
    }
    
    if hasattr(model, 'get_params'):
      params = model.get_params()
      meta['hyperparameters'] = {
        k: v for k, v in params.items()
        if isinstance(v, (int, float, str, bool, type(None))) 
      }
    if extra:
      meta['custom'] = extra
    return meta
    
  
  @staticmethod
  def _fmt_size(size_bytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
      if size_bytes < 1024:
        return f'{size_bytes:.1f} {unit}'
      size_bytes /= 1024
    
    return f'{size_bytes:.1f} TB'