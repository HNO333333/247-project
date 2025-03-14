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
 
### conformer

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

- nfft 8 (conformer_nfft_8.yaml)

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

- bandsep w/ merge_method=concat

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

--- 

- conformer.yaml + window_length=4000 (best test CER!!)

```
{'val_metrics': [{'val/loss': 0.8518169522285461,
                  'val/CER': 13.336287498474121,
                  'val/IER': 2.658396005630493,
                  'val/DER': 1.9051839113235474,
                  'val/SER': 8.772706985473633}],
 'test_metrics': [{'test/loss': 0.8860297203063965,
                   'test/CER': 14.74913501739502,
                   'test/IER': 2.6167819499969482,
                   'test/DER': 2.141003370285034,
                   'test/SER': 9.991349220275879}],
 'best_checkpoint': 'logs/2025-03-07/18-11-55/checkpoints/epoch=130-step=31440.ckpt'}
```

--- 

randomly drop 1 -> 8 channels

```
{'val_metrics': [{'val/loss': 2.0646538734436035, 'val/CER': 31.457687377929688, 'val/IER': 6.1364641189575195, 'val/DER': 2.0381035804748535, 'val/SER': 23.283119201660156}], 'test_metrics': [{'test/loss': 2.2004425525665283, 'test/CER': 31.96188735961914, 'test/IER': 8.380250930786133, 'test/DER': 2.3603291511535645, 'test/SER': 21.2213077545166}], 'best_checkpoint': 'logs/2025-03-07/19-18-32/checkpoints/epoch=145-step=17520.ckpt'}
{'val_metrics': [{'val/loss': 1.9737565517425537, 'val/CER': 29.995569229125977, 'val/IER': 5.582631587982178, 'val/DER': 2.215330123901367, 'val/SER': 22.197607040405273}], 'test_metrics': [{'test/loss': 1.9963494539260864, 'test/CER': 30.727588653564453, 'test/IER': 8.033781051635742, 'test/DER': 2.3170204162597656, 'test/SER': 20.376787185668945}], 'best_checkpoint': 'logs/2025-03-07/21-18-49/checkpoints/epoch=137-step=16560.ckpt'}
{'val_metrics': [{'val/loss': 2.0845963954925537, 'val/CER': 31.214000701904297, 'val/IER': 6.956136226654053, 'val/DER': 2.0159504413604736, 'val/SER': 22.241914749145508}], 'test_metrics': [{'test/loss': 2.2481913566589355, 'test/CER': 32.30835723876953, 'test/IER': 9.67951488494873, 'test/DER': 1.9488955736160278, 'test/SER': 20.679948806762695}], 'best_checkpoint': 'logs/2025-03-07/23-17-40/checkpoints/epoch=131-step=15840.ckpt'}
{'val_metrics': [{'val/loss': 2.030165910720825, 'val/CER': 30.41648292541504, 'val/IER': 5.626938343048096, 'val/DER': 2.303943395614624, 'val/SER': 22.485599517822266}], 'test_metrics': [{'test/loss': 2.184995174407959, 'test/CER': 32.56821060180664, 'test/IER': 9.07319164276123, 'test/DER': 2.598527431488037, 'test/SER': 20.89649200439453}], 'best_checkpoint': 'logs/2025-03-08/01-17-01/checkpoints/epoch=145-step=17520.ckpt'}
{'val_metrics': [{'val/loss': 2.0301311016082764, 'val/CER': 31.67922019958496, 'val/IER': 6.756756782531738, 'val/DER': 2.0602569580078125, 'val/SER': 22.862207412719727}], 'test_metrics': [{'test/loss': 2.162057399749756, 'test/CER': 31.9185791015625, 'test/IER': 8.74837589263916, 'test/DER': 2.273711681365967, 'test/SER': 20.89649200439453}], 'best_checkpoint': 'logs/2025-03-08/03-15-31/checkpoints/epoch=135-step=16320.ckpt'}
{'val_metrics': [{'val/loss': 1.9068659543991089, 'val/CER': 30.70447540283203, 'val/IER': 6.025697708129883, 'val/DER': 1.9716438055038452, 'val/SER': 22.707134246826172}], 'test_metrics': [{'test/loss': 2.047264575958252, 'test/CER': 32.330013275146484, 'test/IER': 8.46686840057373, 'test/DER': 2.1004765033721924, 'test/SER': 21.76266860961914}], 'best_checkpoint': 'logs/2025-03-08/05-13-06/checkpoints/epoch=149-step=18000.ckpt'}
{'val_metrics': [{'val/loss': 2.0890424251556396, 'val/CER': 32.10013198852539, 'val/IER': 6.778910160064697, 'val/DER': 2.1488702297210693, 'val/SER': 23.172351837158203}], 'test_metrics': [{'test/loss': 2.1434166431427, 'test/CER': 33.066261291503906, 'test/IER': 8.16370677947998, 'test/DER': 2.18709397315979, 'test/SER': 22.71546173095703}], 'best_checkpoint': 'logs/2025-03-08/07-09-54/checkpoints/epoch=143-step=17280.ckpt'}
{'val_metrics': [{'val/loss': 1.8927515745162964, 'val/CER': 31.723526000976562, 'val/IER': 6.025697708129883, 'val/DER': 2.259636640548706, 'val/SER': 23.43819236755371}], 'test_metrics': [{'test/loss': 2.0219180583953857, 'test/CER': 32.2433967590332, 'test/IER': 8.271979331970215, 'test/DER': 2.6201817989349365, 'test/SER': 21.351234436035156}], 'best_checkpoint': 'logs/2025-03-08/09-05-49/checkpoints/epoch=122-step=14760.ckpt'}
```

---

downsampling factor: 1.1 -> 1.8

```
{'val_metrics': [{'val/loss': 0.7874718904495239, 'val/CER': 12.871068000793457, 'val/IER': 2.237483501434326, 'val/DER': 1.3291980028152466, 'val/SER': 9.304386138916016}], 'test_metrics': [{'test/loss': 0.9832862615585327, 'test/CER': 15.0498046875, 'test/IER': 2.49025559425354, 'test/DER': 1.6457340717315674, 'test/SER': 10.91381549835205}], 'best_checkpoint': 'logs/2025-03-07/23-22-50/checkpoints/epoch=145-step=17520.ckpt'}
{'val_metrics': [{'val/loss': 0.8546119332313538, 'val/CER': 13.845812797546387, 'val/IER': 2.370403289794922, 'val/DER': 1.5507310628890991, 'val/SER': 9.924678802490234}], 'test_metrics': [{'test/loss': 0.9152058959007263, 'test/CER': 15.699437141418457, 'test/IER': 2.4036378860473633, 'test/DER': 1.7106972932815552, 'test/SER': 11.585102081298828}], 'best_checkpoint': 'logs/2025-03-08/01-12-56/checkpoints/epoch=128-step=15480.ckpt'}
{'val_metrics': [{'val/loss': 0.847457766532898, 'val/CER': 13.779353141784668, 'val/IER': 2.0381035804748535, 'val/DER': 1.5285778045654297, 'val/SER': 10.212671279907227}], 'test_metrics': [{'test/loss': 1.0083379745483398, 'test/CER': 16.738847732543945, 'test/IER': 2.5552186965942383, 'test/DER': 1.9922044277191162, 'test/SER': 12.191425323486328}], 'best_checkpoint': 'logs/2025-03-08/02-54-12/checkpoints/epoch=121-step=14640.ckpt'}
{'val_metrics': [{'val/loss': 0.9095633029937744, 'val/CER': 13.912272453308105, 'val/IER': 2.5254762172698975, 'val/DER': 1.3070447444915771, 'val/SER': 10.079751968383789}], 'test_metrics': [{'test/loss': 1.0829602479934692, 'test/CER': 16.24079704284668, 'test/IER': 2.3603291511535645, 'test/DER': 2.1004765033721924, 'test/SER': 11.779991149902344}], 'best_checkpoint': 'logs/2025-03-08/04-28-47/checkpoints/epoch=137-step=16560.ckpt'}
{'val_metrics': [{'val/loss': 0.8955574035644531, 'val/CER': 14.04519271850586, 'val/IER': 2.5919361114501953, 'val/DER': 1.4399645328521729, 'val/SER': 10.01329231262207}], 'test_metrics': [{'test/loss': 0.9495928883552551, 'test/CER': 15.829363822937012, 'test/IER': 2.6201817989349365, 'test/DER': 2.0571675300598145, 'test/SER': 11.152013778686523}], 'best_checkpoint': 'logs/2025-03-08/05-55-25/checkpoints/epoch=127-step=15360.ckpt'}
{'val_metrics': [{'val/loss': 0.8789817690849304, 'val/CER': 14.466105461120605, 'val/IER': 2.5919361114501953, 'val/DER': 1.7058041095733643, 'val/SER': 10.168365478515625}], 'test_metrics': [{'test/loss': 1.0045500993728638, 'test/CER': 16.00259780883789, 'test/IER': 2.6634907722473145, 'test/DER': 1.6890429258346558, 'test/SER': 11.650065422058105}], 'best_checkpoint': 'logs/2025-03-08/09-19-38/checkpoints/epoch=145-step=17520.ckpt'}
{'val_metrics': [{'val/loss': 0.9014843106269836, 'val/CER': 14.709792137145996, 'val/IER': 2.5476295948028564, 'val/DER': 1.8165706396102905, 'val/SER': 10.34559154510498}], 'test_metrics': [{'test/loss': 0.997227668762207, 'test/CER': 16.089216232299805, 'test/IER': 2.750108242034912, 'test/DER': 2.078821897506714, 'test/SER': 11.260285377502441}], 'best_checkpoint': 'logs/2025-03-08/12-29-24/checkpoints/epoch=147-step=17760.ckpt'}
{'val_metrics': [{'val/loss': 0.9423133730888367, 'val/CER': 15.972530364990234, 'val/IER': 3.0128488540649414, 'val/DER': 1.5064244270324707, 'val/SER': 11.453256607055664}], 'test_metrics': [{'test/loss': 0.9610236287117004, 'test/CER': 16.955392837524414, 'test/IER': 2.5552186965942383, 'test/DER': 1.9922044277191162, 'test/SER': 12.407968521118164}], 'best_checkpoint': 'logs/2025-03-08/13-41-03/checkpoints/epoch=136-step=16440.ckpt'}
```