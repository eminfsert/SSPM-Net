# Track D — Cross-pol'ü Fazla Beslemek: Termal-Farkındalıklı Veri Terimi + Faz Haritalarının Yapısal Kullanımı

## Context

Kullanıcının sorusu: co-pol iyileşirken cross-pol neden zayıf, tasarımdaki şüpheliler ne, faz bu kanalı nasıl besleyebilir? Bu session'da ölçülen cevaplar:

**Neden co-pol iyi, xpol zayıf (kanıtlı şüpheliler):**
1. **Termal taban çarpımsal modeli kırıyor (ana şüpheli).** Piksellerin %8.6'sında HV sinyali termal gürültünün altında (σ_th≈12-14 amp birimi). Kayıp stack'inin üç bileşeni saf çarpımsal speckle varsayar: amplitüd-TV, Rayleigh histogram eşleme, speckle çarpanlara ayırma (y=x·S). Co-pol'de termal ihmal edilebilir → varsayım tutuyor; xpol karanlıkta y=x·S+n afin → ağ termal gürültüyü x'e (parlama) veya S'ye (histogram bozulması) emiyor. Kanıt: GT karanlık-bin göreli RMSE 0.46 (parlakta 0.13); hedef-domain debias sentetikte bias +4.5→+0.6.
2. **Xpol süpervizyonu daha gürültülü ve mevcut w_r bunu ÇÖZMÜYOR.** `merlin_recip_weight` L1 KAYIPLARINI karıştırıyor, ÖLÇÜMLERİ değil — hedef gürültüsünde √2 kazanç yok (w_r taraması bu yüzden düz çıktı). Ölçüldü (sentetik, ρ_speckle~0.8 + bağımsız termal): iki hedefin ortalaması süpervizyon RMSE'sini 7.80→5.74 (−%26) düşürüyor, bias değişmiyor.
3. **Faz bilgisi ağa hiç girmiyor** — sadece elle tasarlanmış skaler kayıp ağırlığı (TV boost, fidelity). Ağın kendisi lokal SNR'ı bilmiyor; polgroup denemesinin grain sorunu da elle ağırlık ayarının kırılganlığını gösterdi — uyarlamayı ağa bırakmak daha sağlam.
4. **Kapanan yol (dürüst negatif, bugün ölçüldü):** koherent HV+VH füzyonu İMKANSIZ — [0,π] katlı faz dosyaları |Re|/|Im| dosyalarıyla piksel bazında TUTARSIZ (kadran-faz korelasyonu 0.02-0.03 ≈ 0; farklı üründen geliyorlar). `sign(Re)=sign(cos φ)` geri kazanımı çöp üretir. Faz yalnızca UZAMSAL İSTATİSTİK haritası olarak kullanılabilir (7×7 koherans haritaları buna dayanıklı — recip haritası 0.655 ile fiziksel).

**Fazın xpol'e verebileceği üç şey (haritalar olarak, kanıtlı):**
- `snr` (HV-VH resiprosite koheransı): zaten var; yeni kullanım = AĞ GİRDİSİ.
- `helix` (YENİ): e^{2i(φ_HH−φ_HV)} lokal koheransı — yansıma simetrisi kıran insan-yapımı hedeflerde yanıyor (parlak-%10'da 0.337 vs rastgele-faz null 0.129; det haritasından güçlü). Xpol'e özel yapı koruyucu.
- Rayleigh-varsayımlı kayıpların floor piksellerinde kapatılması için gate.

## Değişiklikler (öncelik = kanıt gücü sırası; taban = pixel+group + xpol_pair_input)

### D1 — Füzyonlu + hedef-domain debias'lı xpol veri terimi (en güçlü kanıt)
- `trainer.py` MERLIN-L1 dalı: xpol için hedef `t_x = 0.5·(tgt_HV + tgt_VH)` (ölçüm ortalaması; hem d_HV hem d_VH bunu görür; mevcut w_r yolu knob'la korunur: `xpol_fused_target: bool`).
- Hedef-domain termal debias: `t_x' = sqrt(max(t_x² − t·σ_comp², 0))`, σ_comp = σ_th/√2 (bileşen domaini), amp-normalize birimde; knob `xpol_target_debias: float` (0=kapalı). σ_th `estimate_thermal_sigma()` ile (mevcut). Post-hoc debias'ın grain amplifikasyonu problemi burada YOK: ağ+regularizer'lar debias'lı hedefi pürüzsüz öğrenir (sentetik: karanlıkta RMSE 7.23→5.01, bias +4.5→+0.6).
- Not: 2-look ortalama hedefin L1-medyan konvansiyon sabiti hafif kayar — ölçek-bağımsız metrikler etkilenmez; kalibrasyon gerekirse mevcut `_L1_RATIO` mantığıyla düzeltilir.

### D2 — SNR haritası ağ girdisi olarak (`xpol_snr_input`)
- `model.py`: `SSPMNet.forward(x, aux=None)` — xpol branch girdisi (self, recip[, snr]) olur; `Config.xpol_snr_input: bool` (xpol_pair_input altyapısı `in_channels=3`'e zaten genelleşti). aux None ise sıfır düzlemi.
- `trainer.py`/`scene_trainer.py`: model çağrılarına (train + tüm inference/TTA yolları) `aux=snr_t` geçilir.
- Beklenti: ağ karanlık/floor bölgeyi girdiden tanıyıp uyarlanır — elle kayıp ağırlığı yerine öğrenilmiş uzamsal uyarlama.

### D3 — Helix haritası: xpol'e özel yapı koruması (`phase_helix_protect`)
- `phase_data.py`: `phase_feedback_maps`'e `helix` haritası (mean of coh(u_HH·conj(u_HV)), coh(u_HH·conj(u_VH)), robust-norm; null-tabanı 0.129 çıkarılarak normalize).
- `trainer.py`: kanal 1,2 için TV ağırlıklarını ve polish protect'i helix ile artır (yalnız xpol kanallarına; kanal-bazlı tv_weights altyapısı polgroup'tan hazır). Grain dersi: koruma HARİTASI ekleme (smoothing azaltma değil, seçici koruma) — düz alan smoothing'i düşmez.

### D4 — SNR-gate'li çarpanlara-ayırma/histogram (floor'da Rayleigh varsayımı kapansın)
- `trainer.py`: `l_fact` xpol kanallarında piksel-bazlı snr ağırlığı; S histogramı (kanal 1,2) floor piksellerini (snr < eşik) dışlar (`losses.compute_soft_histogram`'a maske parametresi). Knob `fact_snr_gate: float` (0=kapalı).

## Deney protokolü (grain dersi dahil)

- Script `scripts/experiments/run_hv_phase.py`, taban = pixel+group + `xpol_pair_input` (revize kazanan). Varyantlar: taban / D1 / D1+debias-taraması(t=0.5,1.0) / D2 / D3 / D4 / kazanan kombinasyon. Sentetik-GT + gerçek yama, .npy cache.
- **Her tabloya zorunlu yeni sütun: düz-su high-pass std (grain genliği)** — noisy-EPI+ENLr ikilisinin kör noktası kapanır. Su maskesi scripte gömülür (bu session'daki tanım).
- Başarı çıtası: GT PSNR(HV) ≥ +0.3 dB (pairin tabanına karşı) VE su-grain ≤ taban VE gerçek EPI(HV) ≥ taban. Karanlık-bin göreli RMSE ayrıca raporlanır.
- Sentetik protokole termal gürültü zaten dahil (σ_n kalibrasyonlu) — D1 debias'ın GT'si adil ölçülür.
- Her adım commit+push; negatifler dürüstçe arşivlenir; memory+CLAUDE.md checkpoint.

## Dosyalar

- `sspmnet/trainer.py` (D1 hedef füzyon+debias, D2 aux geçişi, D3 xpol protect, D4 gate)
- `sspmnet/model.py`, `sspmnet/config.py` (D2 `xpol_snr_input`, aux forward)
- `sspmnet/phase_data.py` (D3 helix haritası)
- `sspmnet/losses.py` (D4 histogram maskesi)
- `sspmnet/scene_trainer.py` (aynı knob'ların yansıması)
- `scripts/experiments/run_hv_phase.py` (yeni; su-grain sütunlu tablo)

## Doğrulama

1. Smoke: her knob açık/kapalı 6-step mini eğitim + aux'lu forward şekil kontrolü; helix haritası görsel kontrol (parlak yapılarda yanıyor mu — figür).
2. `run_hv_phase.py` tam koşu → `docs/figures/metrics_hv_phase.txt` + HV zoom + su-crop figürleri.
3. Kazanan, pairin tabanı ve base ile aynı tabloda; üç kriterli çıta kontrolü; sonuç ne olursa olsun belgelenir.
