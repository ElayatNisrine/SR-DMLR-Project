"""STEP 22 : embeddings faciaux FaceNet (InceptionResnetV1, VGGFace2) pour ASD Emotions.
Visage detecte et aligne par MTCNN ; si aucun visage n'est detecte, l'image entiere est redimensionnee
(compte rapporte). Meme ordre d'images que step1 (dossiers par emotion, tri alphabetique).
La reduction de dimension (PCA, non supervisee) est faite dans le pipeline : step2 l'ajuste sur tout le pool
(sans etiquette) pour la cohorte ; step27 l'ajuste sur la partie entrainement de chaque seed pour les taches par bloc."""
import glob, json, numpy as np, torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torchvision import transforms

mt = MTCNN(image_size=160, margin=20, post_process=True)
net = InceptionResnetV1(pretrained="vggface2").eval()
fallback = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor(), fixed_image_standardization])
X, lab, miss = [], [], 0
for emo in ["anger", "fear", "joy", "Natural", "sadness", "surprise"]:
    for p in sorted(glob.glob(f"raw/emotion/{emo}/*")):
        if not p.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        im = Image.open(p).convert("RGB")
        face = mt(im)
        if face is None:
            face = fallback(im); miss += 1
        with torch.no_grad():
            X.append(net(face[None]).numpy()[0])
        lab.append(emo)
X = np.array(X, np.float32)
old = np.load("features/emotion.npz", allow_pickle=True)
assert len(X) == len(old["emotion"]) and (np.array(lab) == old["emotion"]).all(), "ordre different de step1"
np.savez_compressed("features/emotion_facenet.npz", X=X, emotion=np.array(lab))
json.dump({"n_images": len(X), "no_face_detected": miss, "dim": int(X.shape[1])}, open("features/emotion_facenet_info.json", "w"))
print(len(X), "images ; sans visage detecte :", miss)
