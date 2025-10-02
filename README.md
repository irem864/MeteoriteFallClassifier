 Proje İçeriği:

meteorite-landings.csv → Meteorite veri seti

proje.py → Tüm veri işleme, model eğitimi ve görselleştirme kodları

.idea/ → PyCharm proje ayar dosyaları

.gitignore → Git için yoksayılacak dosyalar

 Teknolojiler & Kütüphaneler

Python 3.x

Pandas

scikit-learn

Matplotlib

Seaborn

 Projenin Amacı

Meteorite veri setini temizlemek ve ön işleme tabi tutmak

Farklı makine öğrenimi algoritmaları ile "fall" değişkenini tahmin etmek

Modellerin performansını Accuracy, Precision, Recall, F1 Score ve ROC AUC metrikleri ile karşılaştırmak

ROC eğrileri ve karmaşıklık matrisleri ile görselleştirme yapmak

 Kurulum

Depoyu klonlayınız : git clone https://github.com/irem864/MeteoriteFallClassifier.git
                  cd MeteoriteFallClassifier


Gerekli Python kütüphanelerini yükleyiniz:pip install -r requirements.txt




Eğer requirements.txt yoksa manuel yükleyebilirsiniz:
pandas, scikit-learn, matplotlib, seaborn

proje.py dosyasını çalıştırınız:python proje.py


Model Performansı

Projede kullanılan algoritmalar:

SVM (Support Vector Machine)

KNN (K-Nearest Neighbors)

Decision Tree

Performans metrikleri ve görselleştirmeler otomatik olarak oluşturulur:

Bar grafikleri: Accuracy, Precision, Recall, F1 Score, ROC AUC

ROC eğrileri: Tüm modellerin ROC eğrilerini tek grafikte gösterir

Confusion matrix: Her modelin tahmin başarısını ısı haritası olarak gösterir

 Örnek Görselleştirme

ROC Eğrisi

[ ROC eğrisi grafiği burada gösterilecek ]


Confusion Matrix (SVM)

[ Confusion matrix grafiği burada gösterilecek ]

 Kaynaklar

Kaggle Meteorite Landings Dataset

 Lisans

Bu proje MIT lisansı ile korunmaktadır.
