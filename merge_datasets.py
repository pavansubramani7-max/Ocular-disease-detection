import shutil, os, sys

DS1 = r"C:\Users\raksh\Downloads\archive (1) (1)\preprocessed"
DS2 = r"c:\Users\raksh\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\442BAA17E66938FEFD5AAC5C637E379F89A760E4\transfers\2026-15\archive (4)\datasets"
OUT = r"C:\ocular_combined"

MAP1 = {"A":"ageDegeneration","C":"cataract","D":"diabetes",
        "G":"glaucoma","H":"hypertension","M":"myopia","N":"normal"}
MAP2 = {"ageDegeneration":"ageDegeneration","cataract":"cataract",
        "diabetes":"diabetes","glaucoma":"glaucoma","hypertension":"hypertension",
        "myopia":"myopia","normal":"normal"}

for cls in MAP1.values():
    os.makedirs(os.path.join(OUT, cls), exist_ok=True)

def copy_ds(src, fmap, label):
    total = 0
    for folder, target in fmap.items():
        src_dir = os.path.join(src, folder)
        dst_dir = os.path.join(OUT, target)
        if not os.path.exists(src_dir):
            print("MISSING:", src_dir)
            continue
        existing = len(os.listdir(dst_dir))
        count = 0
        for f in os.listdir(src_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ext = os.path.splitext(f)[1]
            shutil.copy2(os.path.join(src_dir, f),
                         os.path.join(dst_dir, label + "_" + str(existing + count) + ext))
            count += 1
        total += count
        print(label, folder, "->", target, ":", count)
        sys.stdout.flush()
    return total

print("Copying Dataset 1...")
t1 = copy_ds(DS1, MAP1, "ds1")
print("Dataset1 total:", t1)

print("Copying Dataset 2...")
t2 = copy_ds(DS2, MAP2, "ds2")
print("Dataset2 total:", t2)

print("\nFinal combined counts:")
grand = 0
for cls in sorted(os.listdir(OUT)):
    n = len(os.listdir(os.path.join(OUT, cls)))
    grand += n
    print("  ", cls, ":", n)
print("  TOTAL:", grand)
