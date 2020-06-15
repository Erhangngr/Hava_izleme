## Sebze Üretiminde Hava Kalitesinin Önemi
Bu projede Çanakkale ve ilçelerinde oluşan zararlı gazların sebze üretimindeki etkisi incelenecektir. 

Doğaya bağlı yapısı gereği tarımsal faaliyetler, iklim değişikliklerinden en çok etkilenen sektörlerden birisidir.

İnsanlar tarafından atmosfere salınan gazların doğal sera gazlarının (CO2,O3,NO2) etkisini arttırması sonucunda yerküre yüzeyinde ortalama sıcaklığın yükselmesi ve meydana gelen iklim değişiklikleri küresel ısınma olarak ifade edilmektedir. 

Türkiye sahip olduğu karmaşık iklim yapısından dolayı, küresel ısınmaya bağlı olarak görülebilecek iklim değişikliğinden en fazla etkilenecek ülkelerden biri olarak görülmektedir. Hava sıcaklıklarındaki ekstrem değişimler sebze türlerinin çiçeklenme dönemleri üzerine de olumsuz etki göstermektedir.

Küresel ısınma ve iklim değişikliği etkilerinin en yoğun görüldüğü tarımsal faaliyet ise çok yıllık bitkiler ani sıcaklık değişimlerinden daha çok etkilendiği için, sebzecilik faaliyetleridir.

Son yıllarda sıklıkla görülen durumlardan en önemlileri ise sebze ağaçlarının kış dinlenme, çiçeklenme, tomurcuk oluşumu ve sebze döneminde meydana gelen ekstrem hava koşullarıdır.

## Veri Seti Oluşturma

Kullanılan veri setleri  Kullanılan veri setleri  

Güney Marmara Kalkınma Ajansı  https://www.gmka.gov.tr/

T.C Tarım ve Orman Bakanlığı 	https://www.tarimorman.gov.tr/

Çevre ve Şehircilik Bakanlığı Hava İzleme   https://www.havaizleme.gov.tr/

Havadaki moleküller µg/m³ cinsinden verilmiştir. Sebze üretimi ise ton cinsinden aylık olarak alınmış olup günlük ortalamaya çevrilmiştir. Yaklaşık 4 aylık bir veri seti ile çalışacağız. Sebze üretimi ön planda olduğu için ve sıcaklık artışından dolayı yaz ayları incelendi.

![](https://github.com/Erhangngr/Hava_izleme/blob/master/%C3%87an/%C3%A7an_excel.PNG)


## Hava Kalite İndeksi

Hava Kalitesi Endeksi, belirli bir yerdeki havanın kalitesinin ifade edilmesi için kullanılan ölçüdür.

Hava Kalitesi ölçümlerinde gösterge sayısının yükselmesi artan hava kirliliği yüzdesinin ciddi sağlık sorunlarına neden olacağını belirtir.

![](https://github.com/Erhangngr/Hava_izleme/blob/master/g%C3%B6rsel/hava_kalitesi.jpg)


## Maksimum ve Minimum Değerlerin Görüldüğü Tarihler


![](https://github.com/Erhangngr/Hava_izleme/blob/master/g%C3%B6rsel/en_k%C3%B6t%C3%BC_deger.PNG)


![](https://github.com/Erhangngr/Hava_izleme/blob/master/%C3%87an/max_min_tarih.PNG)


# Model Oluşturma

## Uzun-Kısa Süreli Bellek (Long Short-Term Memory) 

Recurrent Neural Network çeşidi olan LSTM, yapay sinir ağlarında uzun vadeli bağımlılıkları anlamaya, bağlam farkındalığına sahip sinir ağlarını elde etmeye yarar. 

Temel olarak, düğümler arasında daha karmaşık bir akış mekanizması kullanır.

Back propagation benzeri modellerin yol açtığı hataların logaritmik büyümesi problemini ortadan kaldırıyor.

## Normalizasyon 

Normalizasyon işlemi, makine öğrenmesi için genellikle veri hazırlamanın bir parçası olarak uygulanan bir tekniktir.

Normalleştirmenin amacı, veri kümesindeki sayısal sütunların değerlerini, değerler aralığındaki farklılıkları bozmadan ortak bir ölçeğe uygun biçimde değiştirmektir.

Makine öğrenmesi için, her veri kümesini normalleştirme gerekmeyebilir.

Normalleştirme işlemi, verilerin boyutunu azaltmak veya işlemleri normalleştirilmiş değerlerle uygun aralıklarla gerçekleştirmek ve daha anlamlı ve kolayca yorumlanabilir sonuçlar elde etmek için kullanılabilir. 

## 𝑥𝑦𝑒𝑛𝑖 = 𝑋−𝑋𝑚𝑖𝑛 / 𝑋𝑚𝑎𝑥−𝑋𝑚𝑖𝑛  

#### xyeni: x değişkeni için yeni sayıyı, x: x değişkeni için geçerli sayıyı, xmin: veri setindeki bulunan en küçük sayı, xmax: veri setinde bulunan en büyük sayıyı, ifade etmektedir.

#### Normalizasyon fonksiyonu;

![](https://github.com/Erhangngr/Hava_izleme/blob/master/g%C3%B6rsel/normalizasyon.PNG)

# Model Eğitim

```
import io
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

data = pd.read_excel(io.BytesIO(uploaded['çan_yeni.xlsx']))
data = data.fillna(data.mean())

data

plt.figure(figsize=(16,8))
plt.plot(data.Tarih,data.Sebze)
plt.show()

data = data.filter(['Sebze'])
dataset = data.values
training_data_len = math.ceil( len(dataset) *.8)
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len  , : ]
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
len(train_data)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#LSTM AĞ MODELİ 
model = Sequential()   
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=100)         #modeli eğit

test_data = scaled_data[training_data_len - 60: , : ]    #test veri kümesi

#x ve y test veri kümeleri  oluştur
x_test = []    
y_test =  dataset[training_data_len : , : ] 
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)  #x sayısal değere dönüştürür
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))   # Verileri LSTM tarafından kabul edilen şekle dönüştürün

predictions = model.predict(x_test)    #tahmin işlemleri
predictions = scaler.inverse_transform(predictions)

#rms değerini hesapla
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Tahmin'] = predictions
plt.figure(figsize=(20,8))
plt.title('Model',fontsize=20)
plt.xlabel('Tarih', fontsize=20)
plt.ylabel('Sebze', fontsize=20)
plt.plot(train['Sebze'])
plt.plot(valid[['Sebze', 'Tahmin']])
plt.legend(['Eğitim', 'Veri', 'Tahmin'], loc='lower right')
plt.show()
```

RMSE, yapay sinir ağlarında modellerin performansını değerlendirmek için kullanılan bir indistir. 

Burada ölçüm değerleri ile model tahminleri arasındaki hata oranını belirlemek için kullanılır.

Bu rmse değerinin sıfır olması oluşturulan modelin mükemmel olması demektir.

## Model Başarımları

Model başarımı ve model kaybı arasında ters orantı yakaladığımıza göre başarılı bir model oluşturulmuştur.

Rmse değerinin sıfıra yakınlığı diğer bir kıstasımız olup, gözle görülür bir şekilde sebzelerin kaç bin ton üretildiği gözlenmek istenmektedir.

![](https://github.com/Erhangngr/Hava_izleme/blob/master/g%C3%B6rsel/ba%C5%9Far%C4%B1m2.png)

![](https://github.com/Erhangngr/Hava_izleme/blob/master/g%C3%B6rsel/kay%C4%B1p2.png)

Model oluşturulurken bir çok farklı düğüm ve katman modeli sırayla test edilmiştir. En verimli olan model LSTM olarak belirlendi.

Eğitim için verinin %80'i kullanılmıştır. 60 gerçek veri baz alınarak tahmin gerçekleştirmeye çalışıldı.

Modelin tekrardan eğitilmesi bir yüktür. Her seferinde bu işlemle uğraşmamak için ve bilgisayarımızı yormamak adına bu işlemi gerçekleştirdik. Oluşturduğumuz modelin ağırlık dosyalarını kaydettik. 

# Sebze Üretim Tahmini

Veri setinde Sebze sütunu ton cinsinden verilmiştir. Günlük ilçelerin kaç ton sebze ürettiği bu sütunda mevcut olarak bulunmaktadır.

Yaklaşık 4 aylık bir veri seti ile çalışacağız. Sebze üretimi ön planda olduğu için ve sıcaklık artışından dolayı yaz ayları incelendi.

Ayrıca sera gazlarının yaz aylarında daha çok ön plana çıkması da bilinen bir gerçektir.

![](https://github.com/Erhangngr/Hava_izleme/blob/master/Sebze_Tahmin/genel_sebze.PNG)

## Sebze Üretim Grafiği

Çanakkale il merkezi ve ilçeleri senelik ortalama 977 bin ton sebze üretimi yapmaktadır. 

Günlük üretimin çizgi grafiği de şekilde mevcuttur.

![](https://github.com/Erhangngr/Hava_izleme/blob/master/Sebze_Tahmin/sebze_genel.PNG)

Model eğitiminden sonra ortalama sebze üretimi tahmini çizgi grafiğinde belirtildi. Verilerin %80’lik kısmı eğitim olarak belirlenmişti. 

### Tahminde günlük ortalama 670 ton sebze üretimi yapıldığı ortaya çıkmıştır. 

![](https://github.com/Erhangngr/Hava_izleme/blob/master/Sebze_Tahmin/tahmin.PNG)

![](https://github.com/Erhangngr/Hava_izleme/blob/master/Sebze_Tahmin/sebze_tahmin.PNG)![](https://github.com/Erhangngr/Hava_izleme/blob/master/Sebze_Tahmin/tahmin2.PNG)


Azot dioksit miktarının aşırı dozlara yükselmemesi ortalamalarda kalması, verimin artmasına sebep olmuştur. 

Sebze ve NO2 arasındaki ilişki korelasyon grafiğinde incelenmiştir. 

![](https://github.com/Erhangngr/Hava_izleme/blob/master/Sebze_Tahmin/NO2.png)

Ozon gazının ortalamanın biraz daha  yukarısına çıkması verimlilik açısından ileriki dönemlerde sorun yaratabileceği sinyallerini verdi. 

Burada sebze üretimi normal bir ilerleme göstermiştir.

![](https://github.com/Erhangngr/Hava_izleme/blob/master/Sebze_Tahmin/O3.png)

PM10 gazının düşük seviyelerde ilerlemesi ve verimin de artış gösterdiği gözlemlenmiştir. 

Sebze üretimi için uygun hava koşullarının olduğu söylenebilir.

![](https://github.com/Erhangngr/Hava_izleme/blob/master/Sebze_Tahmin/PM10.png)

Kükürt dioksit gazı da yine PM10 gazına benzer bir artış göstermiştir. Düşük seviyelerde olması tarımın ve üretimin artışına engel olamayacağı anlamına gelmektedir. 

Sebze üretimi için uygun hava koşulu olduğu söylenebilir.

![](https://github.com/Erhangngr/Hava_izleme/blob/master/Sebze_Tahmin/SO2.png)

Hava kalite indeks tablosuna bakarak havanın yeterliliğini belirten bir kod bloğu yazdım. 

Burada çıktı olarak hava kalite indeksinin değerini ve sebze üretimine elverişli olup olmadığını belirlemiş olduk.

```
import io
import pandas as pd
import numpy as np

can_yeni = pd.read_excel(io.BytesIO(uploaded['can_yeni.xlsx']))

istasyon = can_yeni.copy()
istasyon = istasyon.fillna(istasyon.mean())

import warnings
warnings.filterwarnings('ignore')

import aqi 
istasyon_length = len(istasyon)
hki = np.zeros(istasyon_length)
for i in range(istasyon_length):
    me_hki = aqi.to_aqi([
        (aqi.POLLUTANT_PM10, istasyon['PM10'][i])], algo=aqi.ALGO_MEP)
    hki[i] = me_hki

def hava_kalite(x):
    if x <= 50:
        return "İyi"
    elif x <= 100:
        return "Yeterli"
    elif x <= 200:
        return "Orta"
    elif x <= 300:
        return "Kötü"
    elif x <= 400:
        return "Çok Kötü"
    elif x > 400:
        return "Ciddi"

kalite = []
for i in range(120):
    kalite.append(hava_kalite(hki[i]))

hava_kalite_deger = pd.DataFrame(hki)
hava_kalite_deger.columns = ['HKİ']
hava_kalite_deger.head()

kalite_deger = pd.DataFrame(kalite)
kalite_deger.columns = ['HKİ']
```
Hava kalitesi indeksine de bakarak sebze yetiştirmek için uygun olduğunu anlayabiliriz. 

![](https://github.com/Erhangngr/Hava_izleme/blob/master/%C3%87an/hki.PNG)![](https://github.com/Erhangngr/Hava_izleme/blob/master/%C3%87an/hki_deger.PNG)




