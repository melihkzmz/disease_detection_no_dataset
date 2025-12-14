"""
Keras modelini TensorFlow.js formatına dönüştürme scripti
Kullanım: python convert_keras_to_tfjs.py
"""

import tensorflow as tf
import os

def convert_keras_to_tfjs(keras_model_path, output_dir):
    """
    Keras modelini (.keras) TensorFlow.js formatına (.json) dönüştürür
    
    Args:
        keras_model_path: Giriş .keras model dosyası yolu
        output_dir: Çıktı dizini (model.json dosyası buraya kaydedilir)
    """
    try:
        print(f"Model yukleniyor: {keras_model_path}")
        
        # Keras modelini yükle
        model = tf.keras.models.load_model(keras_model_path)
        
        print(f"Model basariyla yuklendi!")
        print(f"Model ozeti:")
        model.summary()
        
        # Çıktı dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nTensorFlow.js formatina donusturuluyor...")
        
        # TensorFlow.js formatına dönüştür
        from tensorflowjs.converters import save_keras_model
        save_keras_model(model, output_dir)
        
        print(f"Model basariyla donusturuldu!")
        print(f"Cikti dizini: {output_dir}")
        print(f"Model dosyasi: {output_dir}/model.json")
        
    except ImportError as ie:
        print(f"HATA: tensorflowjs paketi bulunamadi veya yuklenemedi!")
        print(f"Yuklemek icin: pip install tensorflowjs")
        print(f"Detayli hata: {ie}")
        return False
    except Exception as e:
        print(f"HATA olustu: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # TensorFlow.js import kontrolü
    try:
        import tensorflowjs.converters as tfjs
    except ImportError as e:
        print("HATA: tensorflowjs paketi bulunamadi veya yuklenemedi!")
        print("Yuklemek icin calistirin: pip install tensorflowjs")
        print(f"Detayli hata: {e}")
        exit(1)
    
    # Kemik hastalığı modeli için dönüştürme
    keras_model_path = "./models/bone_disease_model_4class_densenet121_macro_f1.keras"
    output_dir = "./models/bone_disease_model_tfjs"
    
    if not os.path.exists(keras_model_path):
        print(f"HATA: Model dosyasi bulunamadi: {keras_model_path}")
        print("Lutfen model dosyasinin yolunu kontrol edin.")
        exit(1)
    
    print("=" * 60)
    print("Keras -> TensorFlow.js Model Donusturucu")
    print("=" * 60)
    print()
    
    success = convert_keras_to_tfjs(keras_model_path, output_dir)
    
    if success:
        print()
        print("=" * 60)
        print("Donusturme tamamlandi!")
        print("=" * 60)
        print()
        print("HTML dosyasinda model yolu zaten ayarlanmis durumda.")
        print(f"Model konumu: {output_dir}/model.json")
    else:
        print()
        print("=" * 60)
        print("Donusturme basarisiz oldu!")
        print("=" * 60)

