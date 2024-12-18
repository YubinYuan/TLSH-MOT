import numpy as np
import cv2

class LSHIndex:
    def __init__(self, dim, num_tables, key_size):
        self.dim = dim
        self.num_tables = num_tables
        self.key_size = key_size
        self.hash_tables = [dict() for _ in range(num_tables)]
        self.proj_matrices = np.random.randn(num_tables, dim, key_size)
        
    def add(self, feat_vec, obj_id):
        feat_vec = feat_vec.numpy()
        for i in range(self.num_tables):
            proj_vec = self.proj_matrices[i].dot(feat_vec)
            hash_key = self._hash(proj_vec)
            hash_val = self.hash_tables[i].get(hash_key, [])
            hash_val.append(obj_id)
            self.hash_tables[i][hash_key] = hash_val
    
    def query(self, feat_vec, num_results):
        feat_vec = feat_vec.numpy()
        candidates = set()
        for i in range(self.num_tables):
            proj_vec = self.proj_matrices[i].dot(feat_vec)
            hash_key = self._hash(proj_vec)
            hash_val = self.hash_tables[i].get(hash_key, [])
            candidates.update(hash_val)
        
        return list(candidates)[:num_results]
    
    def _hash(self, proj_vec):
        return tuple(np.sign(proj_vec).astype(int))

def track_objects(video_path, first_frame_boxes):
    cap = cv2.VideoCapture(video_path)
    
    # 第一帧的处理
    ret, frame = cap.read()
    first_frame_feats = extract_features(frame, first_frame_boxes)
    lsh_index = LSHIndex(dim=first_frame_feats.shape[1],
                         num_tables=5, 
                         key_size=24)
    for i, feat in enumerate(first_frame_feats):
        lsh_index.add(feat, i)
        
    prev_boxes = first_frame_boxes
    
    while ret:
        candidate_boxes = detect(frame)  # 检测候选区域
        candidate_feats = extract_features(frame, candidate_boxes)
        
        curr_boxes = []
        for feat in candidate_feats:
            obj_ids = lsh_index.query(feat, num_results=1)  
            if obj_ids:
                matched_idx = obj_ids[0]
                curr_boxes.append(candidate_boxes[matched_idx])
            else:
                curr_boxes.append(None)  # 未匹配到已有目标
                
        # 更新LSH索引
        for i, box in enumerate(curr_boxes):
            if box is not None:
                feat = extract_features(frame, [box])[0]
                lsh_index.add(feat, len(prev_boxes)+i)
        
        prev_boxes = [b for b in curr_boxes if b is not None]  # 去除未匹配的框
        
        ret, frame = cap.read()
        
    cap.release()

def extract_features(frame, boxes):
    # 提取目标区域并生成特征
    regions = [frame[y:y+h, x:x+w] for x,y,w,h in boxes]
    region_tensors = [torch.tensor(cv2.resize(r, (224, 224))) 
                      for r in regions]
    region_tensors = torch.stack(region_tensors).float()
    
    with torch.no_grad():
        feats = feature_extractor(region_tensors)
    
    return feats

def detect(frame):
    # 目标检测,返回候选区域
    return candidate_boxes
