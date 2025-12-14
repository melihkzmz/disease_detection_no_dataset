"""
Basit Keras model dönüştürme scripti
TensorFlow.js ile uyumluluk sorunları için alternatif yöntem
"""

import tensorflow as tf
import os

def convert_model():
    """Keras modelini TensorFlow.js formatına dönüştürür"""
    
    keras_model_path = "./models/bone_disease_model_4class_densenet121_macro_f1.keras"
    output_dir = "./models/bone_disease_model_tfjs"
    
    if not os.path.exists(keras_model_path):
        print(f"HATA: Model dosyasi bulunamadi: {keras_model_path}")
        return False
    
    try:
        print("=" * 60)
        print("Keras Model -> TensorFlow.js Donusturucu")
        print("=" * 60)
        print()
        print(f"Model yukleniyor: {keras_model_path}")
        
        # Model yükleme
        model = tf.keras.models.load_model(keras_model_path)
        print("Model basariyla yuklendi!")
        print()
        
        # Model bilgileri
        print("Model Bilgileri:")
        print(f"- Katman sayisi: {len(model.layers)}")
        print(f"- Input shape: {model.input_shape}")
        print(f"- Output shape: {model.output_shape}")
        print()
        
        # Çıktı dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"TensorFlow.js formatina donusturuluyor...")
        print(f"Cikti dizini: {output_dir}")
        print()
        
        # TensorFlow.js formatına dönüştür - tfjs converter kullan
        # Not: tensorflowjs paketinde sorun varsa, komut satırından çalıştırılabilir:
        # tensorflowjs_converter --input_format keras model.keras output_dir
        
        try:
            # Önce yeni API'yi dene
            from tensorflowjs import converters
            converters.save_keras_model(model, output_dir)
            print("DONUSTURME TAMAMLANDI!")
            
        except Exception as e1:
            print(f"Yeni API ile hata: {e1}")
            print("Alternatif yontem deneniyor...")
            
            try:
                # Eski API'yi dene
                import tensorflowjs as tfjs
                tfjs.converters.save_keras_model(model, output_dir)
                print("DONUSTURME TAMAMLANDI!")
                
            except Exception as e2:
                print(f"Eski API ile de hata: {e2}")
                print()
                print("COZUM:")
                print("Komut satirindan asagidaki komutu calistirin:")
                print()
                print(f"tensorflowjs_converter --input_format keras {keras_model_path} {output_dir}")
                print()
                print("Veya backend API kullanin.")
                return False
        
        print()
        print("=" * 60)
        print("Basarili!")
        print("=" * 60)
        print()
        print(f"Model dosyasi: {output_dir}/model.json")
        print("HTML dosyasinda bu yol zaten ayarlanmis durumda.")
        
        return True
        
    except Exception as e:
        print(f"HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    convert_model()


