# Track E — 8-bit Sınırını Kaldırmak: Tam Aralıklı Veri + Log-Domain Eğitim

**Durum:** SIRADAKİ İŞ (2026-09-02'de yazıldı; Track D kapandıktan sonra
başlanacak). Track D'nin dersi: kayıp-yığını knob'ları platoya ulaştı
(A1/A2'den beri her ekleme ≤0.1–0.2 dB ≈ tek-seed gürültüsü). Kalan büyük
kazançlar knob'larda değil, VERİDE ve VERİNİN TEMSİLİNDE.

## Ölçülmüş headroom (neden bu track)

1. **8-bit kırpma: −1.7 / −1.5 dB (HH/HV)** — A4'te eşleştirilmiş uint8
   kırpma protokolüyle ölçüldü (`docs/figures/metrics_a4_sat.txt`). Kanal
   başına piksellerin ~%7–10'u 255'te doymuş; bunlar tam olarak parlak
   yapılar (bina sıraları, stadyum). Bilinen en büyük tek kayıp.
2. Tam aralıklı amplitüd zaten repoda: `data/example_quadpol.npy`
   (float32, aynı pikseller; uint8 kırpma ≈ 274 npy birimi, p99 ≈ 890,
   max ≈ 17k → kanal başına max/p99 ≈ 12x). A4'te bu veri mevcut
   amplitüd-domain pipeline'ını (q99 normalizasyonu + clip(0,5)) düpedüz
   kırdı (ENLr 0.11, EPI(HV) 0.31): ağır parlak kuyruk lineer domainde
   temsil edilemiyor. → Çözüm log-domain.
3. Termal-taban (Track D): hedef-domain debias karanlık bias'ı −%29
   düşürdü ama görsel grain kontrastını artırdı; log-domain'de karanlık
   pikseller doğal olarak daha fazla dinamik aralık alır, debias/grain
   dengesi yeniden ölçülmeli.

## Değişiklikler (öncelik = kanıt gücü / maliyet sırası)

### E1 — Sansürlü (tek-taraflı) veri terimi — MEVCUT VERİYLE HEMEN
- A4'ün "gelecek düzeltme" notu. Doymuş hedef pikselleri (sat maskesi,
  `load_quadpol_tiffs(return_sat=True)`) veri teriminden ÇIKARMAK yerine
  ALT SINIR olarak kullan: `L = |d − t|` yerine doymuş piksellerde
  `L = relu(t_clip − d)` (çıktı kırpma seviyesinin altına inerse ceza,
  üstüne çıkarsa serbest). A4 bulgusu: kırpılmış hedef "biased ama
  bilgili alt sınır"dır; onu tamamen atmak −1.6 dB daha kaybettiriyordu.
- Knob: `TrainConfig.sat_censored: bool` (MERLIN L1 dalında; masked
  dalda da). Regularizer'lar doymuş bölgede TV/NLM'yi düzleştirmesin diye
  guide CV gate zaten var; ek olarak sat bölgesinde TV ağırlığını
  düşür (`sat_tv_relax`).
- Beklenti: parlak top-%1 RMSE düşer, ENL-ROI'deki "düzleşme" kazancı
  kaybolur (dürüst), EPI ↑. Tek başına 1.7 dB'yi geri getirmez (hedef
  bilgisi yok), ama parlak yapı sadakatini artırır.

### E2 — Log-domain pipeline (`TrainConfig.domain = "amp" | "log"`)
- **Girdi/çıktı:** `x_log = log(amp + eps)`; kanal başına robust afin
  normalizasyon (medyan/IQR → ~[−1, 1]); ağ log-yansıtırlığı tahmin
  eder, `denoised = exp(·)`. Tam aralıklı `.npy` amplitüd doğrudan
  kullanılabilir (kuyruk log'da sıkışır).
- **Veri terimleri:** MERLIN L1, log pseudo-amplitüd hedefleriyle
  (`log(k·|Re|)`): log monoton → medyan konvansiyonu KORUNUR (L1
  kalibrasyon sabiti aynen geçer). N2N/blind-spot dalı aynı.
- **Regularizer'lar:** TV/NLM/whiteness log çıktı üzerinde (ratio
  dedektörü farka dönüşür; mevcut log-gradyan kılavuzları zaten bu
  domainde). Polish ve edge_boost log domainde; edge_boost'un dark gate'i
  gereksizleşebilir (ölçülecek).
- **Speckle faktörizasyon + histogram:** `S_log = y_log − d_log`;
  referans histogram log-Rayleigh (`log sqrt(Gamma(L,1/L))`) —
  `compute_reference_histogram(domain="log")`. Bound loss → log aralığı
  için yumuşak sınır.
- **Pre-warmup:** bilateral log-amplitüd üzerinde.
- **Termal debias:** yoğunluk domaininde (`exp(2·d_log) − k·σ²`) sonra
  log'a dön; sadece xpol.
- Faz haritaları (snr) değişmez.

### E3 — Tam aralıklı RI (kullanıcı verisi gelince)
- Tam aralıklı float |Re|/|Im| (veya imzalı Re/Im) → `calibrate_ri`
  tam aralıklı amplitüde; sat maskesi tamamen kalkar; E1 gereksizleşir.
- İmzalı float SLC gelirse: gerçek MERLIN (koherent), C3 kovaryans
  yolu — ayrı track (gelecek çalışma).

## Deney protokolü

- Script: `scripts/experiments/run_track_e.py`, `.npy` cache'li, aynı
  seed/gürültü altyapısı (`run_hv_phase.py`).
- **Sentetik-GT (tavan ölçümü):** clean proxy TAM ARALIKLI veriden;
  speckle+termal simülasyonu kırpmasız. Satırlar: (a) uint8-kırpmalı
  giriş + amp pipeline (A4 kontrolü, bugünkü durum), (b) uint8 + E1,
  (c) tam aralık + log pipeline (E2), (d) tam aralık + log + E1'siz
  (kırpma yok). (a)→(c) farkı = geri kazanılan dB.
- **Gerçek yama:** tam aralıklı `.npy` ile E2; ENLr, EPI, corr, su
  grain sütunları + YENİ: parlak top-%1 sadakat (kırpma seviyesinin
  üstündeki piksellerde çıktı/giriş oranı; düzleşme dedektörü).
- **Çıta:** GT'de kırpma maliyetinin ≥ +1.0 dB'si geri (her iki kanal),
  eşit EPI, su grain ≤ base; gerçekte parlak yapılar düzleşmeden
  (top-%1 oranı ≥ 0.9) ENLr ≈ 1.
- Tek-seed gürültü tabanı (Track D'de ölçülen base@s43 farkı) her
  tabloya not düşülür; finalistler ikinci seed'le doğrulanır.

## Sıra ve tahmini maliyet

1. E1 sansürlü kayıp — küçük değişiklik, mevcut veri: 1 session.
2. E2 log-domain — trainer/losses/scene_trainer yeniden düzenleme:
   ~2 session (smoke + tavan ölçümü dahil).
3. E3 + Track B (sahne yamaları) — kullanıcı verisine bağlı; Track B
   makinesi hazır (`denoise_scene`, sanity PASS), E2 üstünde koşar.

## Dosyalar

- `sspmnet/trainer.py` (domain anahtarı, sansürlü kayıp, log debias)
- `sspmnet/losses.py` (log-Rayleigh referans, sansürlü L1, log-domain
  bound), `sspmnet/complex_data.py` (tam aralıklı yükleyici; `.npy` +
  RI yeniden kalibrasyon zaten `load_scene_patches`'te var)
- `sspmnet/scene_trainer.py` (aynı anahtarların yansıması)
- `scripts/experiments/run_track_e.py`

## Kullanıcıdan beklenenler

1. **Sahne yamaları (Track B):** ~16 adet 512×512, `data/scene/` içine,
   `{prefix}{pol}_{comp}.tiff` adlandırması (amp/real/imgy/pha ×
   hh/hv/vh/vv); mümkünse her yama için tam aralıklı `{prefix}amp.npy`.
2. **Tam aralıklı Re/Im:** mevcut yama (ve sahne yamaları) için float
   |Re|/|Im| — imzalı Re/Im olursa çok daha iyi (koherent yol açılır).
   `example_quadpol.npy`'nin geldiği üründen aynı piksellerin Re/Im'i.
3. **Teslimat önceliği kararı:** radyometrik doğruluk (termal debias
   açık, k≈1–1.5; karanlık bias −%29 ama düz-alan grain kontrastı ↑) mi,
   görsel temizlik (debias kapalı) mi? Varsayılan buna göre belirlenir.
4. (İsteğe bağlı) düz/kırsal bir yama — `phase_surface_boost` ve
   log-domain'in düz sahnedeki davranışı için.
