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