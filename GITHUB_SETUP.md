# GitHub'a Yükleme ve Ortak Çalışma Rehberi

## 1. GitHub'da Yeni Repository Oluşturma

1. GitHub.com'a giriş yapın
2. Sağ üst köşedeki **"+"** butonuna tıklayın ve **"New repository"** seçin
3. Repository bilgilerini doldurun:
   - **Repository name**: `disease_detection_no_dataset` (veya istediğiniz isim)
   - **Description**: "Multi-disease detection system with ML models for skin, eye, lung, and bone diseases"
   - **Visibility**: Public veya Private seçin
   - **⚠️ ÖNEMLİ**: "Initialize this repository with a README" seçeneğini **İŞARETLEMEYİN** (zaten kodunuz var)
4. **"Create repository"** butonuna tıklayın

## 2. Projeyi GitHub'a Yükleme

GitHub'da repository oluşturduktan sonra, aşağıdaki komutları terminalde çalıştırın:

```bash
# GitHub repository URL'inizi ekleyin (örnek: https://github.com/kullaniciadi/disease_detection_no_dataset.git)
git remote add origin https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git

# Ana branch'i main olarak değiştirin (GitHub'ın yeni standardı)
git branch -M main

# Projeyi GitHub'a yükleyin
git push -u origin main
```

**Not**: `KULLANICI_ADINIZ` ve `REPO_ADINIZ` kısımlarını kendi GitHub bilgilerinizle değiştirin.

## 3. Ortak Çalışma İçin Ayarlar

### 3.1. Collaborator Ekleme (Özel Repository için)

1. GitHub repository sayfanıza gidin
2. **Settings** sekmesine tıklayın
3. Sol menüden **Collaborators** seçin
4. **"Add people"** butonuna tıklayın
5. Çalışmak istediğiniz kişinin GitHub kullanıcı adını veya email adresini girin
6. Kişiye davet gönderin

### 3.2. Ortak Çalışma İçin Git Komutları

#### Yeni Değişiklikleri Çekme (Pull)
```bash
# Uzaktaki değişiklikleri çekin
git pull origin main
```

#### Değişiklikleri Yükleme (Push)
```bash
# Değişiklikleri stage'e ekleyin
git add .

# Commit yapın
git commit -m "Değişiklik açıklaması"

# GitHub'a yükleyin
git push origin main
```

#### Yeni Branch Oluşturma (Özellik Geliştirme için)
```bash
# Yeni branch oluştur ve geçiş yap
git checkout -b feature/yeni-ozellik

# Değişiklikleri yap, commit et
git add .
git commit -m "Yeni özellik eklendi"

# Branch'i GitHub'a yükle
git push origin feature/yeni-ozellik
```

### 3.3. Pull Request (PR) Oluşturma

1. GitHub repository sayfanıza gidin
2. **"Pull requests"** sekmesine tıklayın
3. **"New pull request"** butonuna tıklayın
4. Base branch: `main`, Compare branch: `feature/yeni-ozellik` seçin
5. Değişiklikleri gözden geçirin ve **"Create pull request"** tıklayın
6. PR açıklaması ekleyin ve review için işaretleyin

## 4. Ortak Çalışma İçin İpuçları

### 4.1. Commit Mesajları İçin Best Practices
- Açıklayıcı commit mesajları yazın
- Örnek: `"Add bone disease classification model"` ✅
- Örnek: `"Fix eye disease API bug"` ✅
- Örnek: `"Update"` ❌ (çok belirsiz)

### 4.2. Conflict Çözümü
Eğer aynı dosyada farklı değişiklikler yapıldıysa:
```bash
# Önce uzaktaki değişiklikleri çekin
git pull origin main

# Conflict varsa, dosyaları düzenleyin ve:
git add .
git commit -m "Merge conflicts resolved"
git push origin main
```

### 4.3. .gitignore Dosyası
Projenizde `.gitignore` dosyası zaten oluşturuldu. Bu dosya:
- Büyük model dosyalarını (.keras, .h5)
- Dataset dosyalarını
- Log dosyalarını
- Virtual environment klasörlerini
- IDE ayarlarını

GitHub'a yüklenmesini engeller.

## 5. GitHub Actions (CI/CD) - İsteğe Bağlı

Otomatik test ve deployment için `.github/workflows/` klasörü oluşturabilirsiniz.

## 6. Issues ve Project Management

- **Issues**: Hata bildirimi ve özellik istekleri için kullanın
- **Projects**: Proje yönetimi için Kanban board oluşturun
- **Milestones**: Versiyon planlaması için kullanın

## 7. Hızlı Komutlar Özeti

```bash
# Durum kontrolü
git status

# Değişiklikleri görmek
git diff

# Commit geçmişi
git log

# Branch listesi
git branch

# Remote repository bilgisi
git remote -v
```

## Sorun Giderme

### "Permission denied" hatası alıyorsanız:
- GitHub'da Personal Access Token oluşturun
- Token'ı şifre yerine kullanın

### "Repository not found" hatası alıyorsanız:
- Repository URL'ini kontrol edin
- Repository'nin var olduğundan emin olun
- Erişim izinlerinizi kontrol edin

---

**İyi çalışmalar! 🚀**

