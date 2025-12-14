#!/bin/bash
# Proje dizinini bulma scripti

echo "======================================"
echo "🔍 PROJE DIZINI BULUCU"
echo "======================================"
echo ""

# Olası dizin yolları
POSSIBLE_PATHS=(
    "/mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier"
    "/mnt/c/Users/melih/Desktop/disease_detection/Skin-Disease-Classifier"
    "/mnt/c/Users/melih/Documents/disease_detection/Skin-Disease-Classifier"
    "/home/melih/dev/disease_detection/Skin-Disease-Classifier"
    "/home/melih/disease_detection/Skin-Disease-Classifier"
    "$HOME/dev/disease_detection/Skin-Disease-Classifier"
    "$HOME/disease_detection/Skin-Disease-Classifier"
    "$(pwd)/Skin-Disease-Classifier"
)

echo "Aranan proje dizini: Skin-Disease-Classifier/"
echo ""

# Her yol için kontrol et
FOUND=false
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✅ BULUNDU: $path"
        echo ""
        echo "Bu dizine gitmek için:"
        echo "  cd $path"
        echo ""
        
        # Script var mı kontrol et
        if [ -f "$path/train_bone_4class_macro_f1.py" ]; then
            echo "✅ Eğitim scripti bulundu!"
        fi
        
        # Dataset var mı kontrol et
        if [ -d "$path/datasets/bone/Bone_4Class_Final" ]; then
            echo "✅ Dataset bulundu!"
        fi
        
        FOUND=true
        break
    fi
done

if [ "$FOUND" = false ]; then
    echo "❌ Proje dizini otomatik bulunamadı."
    echo ""
    echo "Manuel arama yapılıyor..."
    echo ""
    
    # Daha geniş arama
    echo "Windows dizinlerinde aranıyor (/mnt/c/Users/...)..."
    if [ -d "/mnt/c/Users/melih" ]; then
        RESULT=$(find /mnt/c/Users/melih -type d -name "Skin-Disease-Classifier" 2>/dev/null | head -1)
        if [ ! -z "$RESULT" ]; then
            echo "✅ BULUNDU: $RESULT"
            echo ""
            echo "Bu dizine gitmek için:"
            echo "  cd $RESULT"
        else
            echo "❌ Bulunamadı"
        fi
    fi
    
    echo ""
    echo "Home dizininde aranıyor ($HOME/...)..."
    if [ -d "$HOME" ]; then
        RESULT=$(find "$HOME" -type d -name "Skin-Disease-Classifier" 2>/dev/null | head -1)
        if [ ! -z "$RESULT" ]; then
            echo "✅ BULUNDU: $RESULT"
            echo ""
            echo "Bu dizine gitmek için:"
            echo "  cd $RESULT"
        else
            echo "❌ Bulunamadı"
        fi
    fi
fi

echo ""
echo "======================================"
echo "Manuel arama komutları:"
echo "======================================"
echo ""
echo "# Windows dizinlerinde ara:"
echo "find /mnt/c/Users/melih -type d -name 'Skin-Disease-Classifier' 2>/dev/null"
echo ""
echo "# Home dizininde ara:"
echo "find ~ -type d -name 'Skin-Disease-Classifier' 2>/dev/null"
echo ""
echo "# Tüm sistemde ara (yavaş):"
echo "find /mnt/c -type d -name 'Skin-Disease-Classifier' 2>/dev/null"
echo ""

