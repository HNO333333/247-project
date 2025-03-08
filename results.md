### baseline:
- 150 epochs, 32 batch size

```
{'val_metrics': [{'val/loss': 1.0054988861083984,
                  'val/CER': 18.941072463989258,
                  'val/IER': 5.361098766326904,
                  'val/DER': 1.3956578969955444,
                  'val/SER': 12.18431568145752}],
 'test_metrics': [{'test/loss': 1.1295205354690552,
                   'test/CER': 21.958072662353516,
                   'test/IER': 5.100497245788574,
                   'test/DER': 2.1612274646759033,
                   'test/SER': 14.6963472366333}],
 }
 ```
 
### whisper_base:
- 150 epochs, 32 batch size
```
{'val_metrics': [{'val/loss': 1.0051476955413818,
              'val/CER': 19.29552459716797,
              'val/IER': 5.338945388793945,
              'val/DER': 1.4178112745285034,
              'val/SER': 12.53876781463623}],
'test_metrics': [{'test/loss': 1.130611777305603,
               'test/CER': 22.36870574951172,
               'test/IER': 4.473741054534912,
               'test/DER': 2.3341257572174072,
               'test/SER': 15.56083869934082}],
}
 ```
 
### baseline - nk_clean
- 150 epochs, 32 batch size
```
{'val_metrics': [{'val/loss': 1.0014753341674805,
                  'val/CER': 18.80815315246582,
                  'val/IER': 6.003544330596924,
                  'val/DER': 1.373504638671875,
                  'val/SER': 11.431103706359863}],
 'test_metrics': [{'test/loss': 1.1928857564926147,
                   'test/CER': 22.282257080078125,
                   'test/IER': 6.289172172546387,
                   'test/DER': 1.404797911643982,
                   'test/SER': 14.588286399841309}],
}
 ```
 
### conformer (test is windowed, need more work)

- conformer.yaml

```
{'val_metrics': [{'val/loss': 0.7741119861602783,
                  'val/CER': 12.472308158874512,
                  'val/IER': 2.0159504413604736,
                  'val/DER': 1.2627381086349487,
                  'val/SER': 9.193619728088379}],
 'test_metrics': [{'test/loss': 0.9087711572647095,
                   'test/CER': 15.093113899230957,
                   'test/IER': 2.381983518600464,
                   'test/DER': 1.7106972932815552,
                   'test/SER': 11.000432968139648}],
 'best_checkpoint': './logs/conformer-best/checkpoints/epoch=136-step=16440.ckpt'}
```

--- 

- crop 1:17 frequecy bins (conformer_log_16.yaml)

```
{'val_metrics': [{'val/loss': 0.8458750247955322,
                  'val/CER': 13.579973220825195,
                  'val/IER': 2.1267168521881104,
                  'val/DER': 1.3513513803482056,
                  'val/SER': 10.10190486907959}],
 'test_metrics': [{'test/loss': 0.9526205658912659,
                   'test/CER': 15.179731369018555,
                   'test/IER': 2.2087483406066895,
                   'test/DER': 1.9488955736160278,
                   'test/SER': 11.022087097167969}],
 'best_checkpoint': 'logs/2025-03-06/10-45-14/checkpoints/epoch=147-step=17760.ckpt'}
```

--- 

- downsample 2x (conformer_ds.yaml)

```
{'val_metrics': [{'val/loss': 0.9792105555534363,
              'val/CER': 16.592823028564453,
              'val/IER': 3.1014621257781982,
              'val/DER': 1.5728843212127686,
              'val/SER': 11.918476104736328}],
'test_metrics': [{'test/loss': 1.0976539850234985,
               'test/CER': 16.80381202697754,
               'test/IER': 2.9233434200286865,
               'test/DER': 2.3170204162597656,
               'test/SER': 11.563446998596191}],
'best_checkpoint': 'logs/2025-03-06/13-25-00/checkpoints/epoch=135-step=16320.ckpt'}
```

---

- bandsep (conformer_bandsep.yaml)
  - add

```
{'val_metrics': [{'val/loss': 1.0244234800338745,
                  'val/CER': 17.235267639160156,
                  'val/IER': 3.1236155033111572,
                  'val/DER': 2.3925564289093018,
                  'val/SER': 11.719096183776855}],
 'test_metrics': [{'test/loss': 1.176459789276123,
                   'test/CER': 19.857080459594727,
                   'test/IER': 2.7067995071411133,
                   'test/DER': 2.836725950241089,
                   'test/SER': 14.313555717468262}],
 'best_checkpoint': 'logs/2025-03-06/16-37-24/checkpoints/epoch=137-step=16560.ckpt'}
```

---

- nfft 8 (conformer_nfft_8.yaml) (currently running at screen 247)

```
{'val_metrics': [{'val/loss': 1.032233476638794,
                  'val/CER': 18.941072463989258,
                  'val/IER': 3.4780681133270264,
                  'val/DER': 1.1076650619506836,
                  'val/SER': 14.355339050292969}],
 'test_metrics': [{'test/loss': 1.1373990774154663,
                   'test/CER': 20.723257064819336,
                   'test/IER': 3.9411001205444336,
                   'test/DER': 1.4075356721878052,
                   'test/SER': 15.374621391296387}],
 'best_checkpoint': 'logs/2025-03-07/13-35-28/checkpoints/epoch=131-step=15840.ckpt'}

```

---

- bandsep w/ merge_method=concat (currently running at screen concat)

```
{'val_metrics': [{'val/loss': 1.0953184366226196,
                  'val/CER': 19.605670928955078,
                  'val/IER': 2.4368631839752197,
                  'val/DER': 5.117412567138672,
                  'val/SER': 12.051395416259766}],
 'test_metrics': [{'test/loss': 1.2882496118545532,
                   'test/CER': 21.351234436035156,
                   'test/IER': 2.7717626094818115,
                   'test/DER': 4.287570476531982,
                   'test/SER': 14.291901588439941}],
 'best_checkpoint': 'logs/2025-03-07/14-12-29/checkpoints/epoch=127-step=15360.ckpt'}
```

