# Simple Bone Cyst (Basit Kemik Kisti) - Açıklama

## 🦴 Simple Bone Cyst Nedir?

**Simple Bone Cyst** (SBC) veya **Unicameral Bone Cyst**, kemik içinde sıvı dolu bir boşluk (kist) olan bir durumdur.

---

## 📋 Özellikler

### Tıbbi Tanım:
- **İyi huylu (benign)** bir durumdur
- **Tümör DEĞİLDİR** (kanser değil)
- Kemik içinde sıvı dolu bir boşluk
- Genellikle **asemptomatik** (belirti göstermez)

### Lokalizasyon:
- Genellikle **uzun kemiklerde** görülür (humerus, femur)
- Çocuk ve genç erişkinlerde daha sık
- Erkeklerde kadınlardan daha yaygın

### Görünüm:
- X-ray'de **yuvarlak/oval şeffaf alan**
- Kemik korteksinde incelme
- Genellikle tek odaklı

---

## 🔍 Veri Setindeki Yeri

### Mevcut 5 Sınıflı Modelde:
- **Simple_Bone_Cyst** - 206 görüntü
  - Train: 164
  - Val: 20
  - Test: 22

### Neden Ayrı Sınıf?
1. **Tümör değil** - Benign_Tumor'a dahil edilmedi
2. **Kist** - Tümörlerden farklı bir patoloji
3. **Tıbbi önem** - Tümörlerle karıştırılmamalı

---

## 💡 Model Açısından

### Senaryo 1: Mevcut (Ayrı Tutuldu)
```
5 Sınıf:
- Normal
- Fracture
- Benign_Tumor
- Malignant_Tumor
- Simple_Bone_Cyst ✅ (Ayrı)
```

**Avantaj:**
- Tıbbi açıdan doğru (kist ≠ tümör)
- Ayrı tanımlanabilir

**Dezavantaj:**
- Küçük sınıf (164 train)
- Sınıf sayısı 5

---

### Senaryo 2: Benign_Tumor'a Dahil Et

```
4 Sınıf:
- Normal
- Fracture
- Benign_Tumor (Simple_Bone_Cyst dahil)
- Malignant_Tumor
```

**Avantaj:**
- Daha az sınıf (4)
- Daha yüksek accuracy beklenir
- Simple_Bone_Cyst daha büyük gruba dahil (164 + 1,054 = 1,218)

**Dezavantaj:**
- Tıbbi açıdan yanıltıcı (kist ≠ tümör)
- Ama pratik açıdan kabul edilebilir (ikisi de benign/non-malignant)

---

## 🤔 Öneriler

### Seçenek A: Mevcut (5 Sınıf) - ÖNERİLEN
- ✅ Tıbbi açıdan doğru
- ✅ Kist ve tümör ayrımı var
- ⚠️ Küçük sınıf ama yeterli (164 train)

### Seçenek B: Benign_Tumor'a Dahil (4 Sınıf)
- ✅ Daha az sınıf
- ✅ Daha yüksek accuracy
- ⚠️ Tıbbi açıdan yanıltıcı (ama pratik kullanım için kabul edilebilir)

---

## 🏥 Tıbbi Önem

**Simple Bone Cyst:**
- İyi huylu (benign)
- Genellikle tedavi gerektirmez
- Çoğu zaman sadece takip edilir
- Kanser değildir

**Benign_Tumor (Osteochondroma, vb.):**
- İyi huylu tümör
- Genellikle benign ama tümör
- Kistten farklı bir patoloji

**Sonuç:**
- Tıbbi açıdan **farklı kategoriler**
- Ancak **ikisi de benign** (zararsız)

---

## 🎯 Karar

**Mevcut durum (5 sınıf):**
- Simple_Bone_Cyst **ayrı tutuldu**
- Tıbbi açıdan daha doğru

**Alternatif (4 sınıf):**
- Simple_Bone_Cyst → Benign_Tumor'a dahil edilebilir
- Daha yüksek accuracy beklenir
- Pratik kullanım için kabul edilebilir

**Hangi yaklaşımı tercih edersin?**

