import os

root = r"D:\FYP\Processed_MRI"
count_valid = 0
for pid in os.listdir(root):
    path = os.path.join(root, pid, "Axial", "T2")
    if os.path.exists(path):
        num_imgs = len([f for f in os.listdir(path) if f.endswith(".png")])
        if num_imgs >= 16:
            count_valid += 1
print("✅ Patients with >=16 slices:", count_valid)
