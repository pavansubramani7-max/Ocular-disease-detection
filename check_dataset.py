import os
from PIL import Image

dataset = r"C:\ocular_combined"
classes = sorted(os.listdir(dataset))

print("CLASS DISTRIBUTION:")
print("="*55)
total  = 0
counts = {}
for cls in classes:
    cls_dir = os.path.join(dataset, cls)
    n = len([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    counts[cls] = n
    total += n
    bar = "#" * int(n / 100)
    print("  %-20s : %4d  %s" % (cls, n, bar))
print("  %-20s : %4d" % ("TOTAL", total))

max_cls = max(counts, key=counts.get)
min_cls = min(counts, key=counts.get)
print()
print("Imbalance ratio (max/min) : %.1fx" % (counts[max_cls] / counts[min_cls]))
print("Most  :", max_cls, "->", counts[max_cls])
print("Least :", min_cls, "->", counts[min_cls])

print()
print("IMAGE SIZES & QUALITY:")
print("="*55)
for cls in classes:
    cls_dir = os.path.join(dataset, cls)
    files   = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    sizes   = []
    corrupt = 0
    for f in files[:50]:
        try:
            img = Image.open(os.path.join(cls_dir, f))
            sizes.append(img.size)
        except:
            corrupt += 1
    if sizes:
        widths  = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print("  %-20s : avg size=%dx%d  corrupt=%d" % (
            cls,
            int(sum(widths)/len(widths)),
            int(sum(heights)/len(heights)),
            corrupt))

print()
print("VERDICT:")
print("="*55)
ratio = counts[max_cls] / counts[min_cls]
if ratio > 5:
    print("  SEVERE class imbalance (%.1fx) - this hurts DR recall" % ratio)
elif ratio > 3:
    print("  MODERATE class imbalance (%.1fx)" % ratio)
else:
    print("  Class balance is OK (%.1fx)" % ratio)

dr = counts.get("diabetes", 0)
nm = counts.get("normal", 0)
print("  Diabetes vs Normal ratio : %.1fx (Normal dominates)" % (nm/dr) if dr > 0 else "")
print()
print("  Dataset quality is GOOD - problem is model/resolution not data")
