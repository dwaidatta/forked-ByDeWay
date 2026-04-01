
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class SpatialAnalyzer:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the SpatialAnalyzer with a YOLO model.
        Args:
            model_path (str): Path or name of the YOLO model to load.
        """
        print(f"Initializing SpatialAnalyzer with model='{model_path}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        print("SpatialAnalyzer initialized.")

    def analyze(self, image_array, masks, max_relations_per_layer: int = 6):
        """
        Analyze the image and return spatial relationships between objects within the same depth layer.
        
        Args:
            image_array (np.array): Determine objects from this image (H, W, 3).
            masks (list): List of 3 boolean/binary masks [top, bottom, mid] corresponding to 
                          [Closest, Farthest, Mid Range].
                          Shape of each mask: (H, W, 1).
        
        Returns:
            list: A list of 3 strings, each containing the spatial description for the corresponding layer.
                  e.g., ["Object A is to the left of Object B...", "", ""]
        """
        # Run object detection
        results = self.model(image_array, verbose=False, stream=False)
        result = results[0]
        
        full_descriptions = ["", "", ""] # Closest, Farthest, Mid
        
        # Categorize objects into layers
        layer_objects = [[], [], []] # Closest, Farthest, Mid
        
        # Get class names
        names = result.names
        
        if not result.boxes:
             return full_descriptions

        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls]
            
            # Calculate center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Determine which layer this object belongs to
            
            best_layer = -1
            max_overlap = -1
            
            box_area = (x2 - x1) * (y2 - y1)
            if box_area <= 0: continue

            for i, mask in enumerate(masks):
                # mask is (H, W, 1) usually boolean or 0/1
                # Extract mask crop for the box
                # Ensure coordinates are within bounds
                h, w = mask.shape[:2]
                bx1, by1 = max(0, x1), max(0, y1)
                bx2, by2 = min(w, x2), min(h, y2)
                
                if bx2 <= bx1 or by2 <= by1:
                    continue
                
                mask_crop = mask[by1:by2, bx1:bx2]
                overlap_count = np.count_nonzero(mask_crop)
                
                if overlap_count > max_overlap:
                    max_overlap = overlap_count
                    best_layer = i

            # Threshold for assignment? If max_overlap is very small, maybe ignore?
            # For now, assign to best layer if at least some overlap.
            if max_overlap > 0:
                layer_objects[best_layer].append({
                    "label": label,
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "conf": conf
                })

        # Generate spatial descriptions for each layer
        for i in range(3):
            objs = layer_objects[i]
            if len(objs) < 2:
                continue
            
            # Add unique identifiers if multiple objects of same class exist
            # Count occurrences first
            label_counts = {}
            for obj in objs:
                label = obj["label"]
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Assign indices
            current_counts = {}
            for obj in objs:
                label = obj["label"]
                if label_counts[label] > 1:
                    idx = current_counts.get(label, 0) + 1
                    current_counts[label] = idx
                    obj["display_label"] = f"{label} {idx}"
                else:
                    obj["display_label"] = label
            
            # Sort objects from left to right to make descriptions more natural?
            # Or just pair them?
            # "Object A (left) is to the left of Object B (right)"
            
            objs.sort(key=lambda x: x["center"][0]) # Sort by X coordinate
            
            layer_desc = []
            
            for j in range(len(objs)):
                for k in range(j + 1, len(objs)):
                    obj_a = objs[j]
                    obj_b = objs[k]
                    
                    label_a = obj_a["display_label"]
                    label_b = obj_b["display_label"]
                    
                    # Logic for Left/Right
                    # Since sorted by X: A is always to the left of B (mostly)
                    # We can check vertical alignment too.
                    
                    wa = obj_a["box"][2] - obj_a["box"][0]
                    wb = obj_b["box"][2] - obj_b["box"][0]
                    ha = obj_a["box"][3] - obj_a["box"][1]
                    hb = obj_b["box"][3] - obj_b["box"][1]
                    
                    # Use a margin to decide if "Above"/"Below" or just "Left"/"Right"
                    # If vertical overlap is significant, then Left/Right.
                    # If separate vertically, then Above/Below.
                    
                    # Vertical overlap
                    y_overlap = min(obj_a["box"][3], obj_b["box"][3]) - max(obj_a["box"][1], obj_b["box"][1])
                    min_h = min(ha, hb)
                    
                    relation = ""
                    
                    # Heuristic: if y_overlap > 0.5 * min_h, they are side-by-side
                    if y_overlap > 0.5 * min_h:
                        # Side by side
                        if obj_a["center"][0] < obj_b["center"][0]:
                             relation = f"{label_a} is to the left of {label_b}"
                        else:
                             relation = f"{label_a} is to the right of {label_b}"
                    else:
                        # Above/Below
                        if obj_a["center"][1] < obj_b["center"][1]:
                            relation = f"{label_a} is above {label_b}"
                        else:
                            relation = f"{label_a} is below {label_b}"
                    
                    # Avoid duplicates or redundant info? 
                    # "A left of B", "B right of A". Just A left of B is enough.
                    
                    # Also, if multiple same-class objects, might be confusing ("cup is left of cup")
                    # We can add index or position hint?
                    # For now keep it simple as requested.
                    
                    layer_desc.append(relation)
            
            # Cap relations to avoid prompt blow-up (pairwise relations grow O(n^2)).
            if max_relations_per_layer is not None and max_relations_per_layer > 0:
                layer_desc = layer_desc[:max_relations_per_layer]
            full_descriptions[i] = ". ".join(layer_desc)
            
        return full_descriptions
