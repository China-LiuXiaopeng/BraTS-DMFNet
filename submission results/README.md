Please follow the instructions to reproduce the experimental results from the provided "model_last.pth" file, which are 80.12%, 90.62% and 84.54% for Dice_ET, Dice_WT and Dice_TC respectively.

If you make inference using the "model_last.pth" ckpt, the following details may helpful to you to reproduce the results.

(1) In "preprocess.py", find the code and make sure they are using the number of 0.2 and 99.8.

        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)


(2) Check the data normalization! The normalization with float32 or float64 have a significantly impact on it, make a difference in the final prediction. I select one sample of the pkl,  path = "./HGG/Brats18_2013_2_1/Brats18_2013_2_1_data_f32.pkl. You could write a script to print the data. For example,
    
        def debug():
          path = "./HGG/Brats18_2013_2_1/Brats18_2013_2_1_data_f32.pkl"
          with open(path,'rb') as f:
              data,label = pickle.load(f)
          print(data) 
          print(np.min(data), np.max(data))

  It will print like this:

    [[[[-2.8575273 -2.4078138 -3.4921865 -2.6610713]
     [-2.8575273 -2.4078138 -3.4921865 -2.6610713]
     [-2.8575273 -2.4078138 -3.4921865 -2.6610713]
     ...
     ...
     [-2.8575273 -2.4078138 -3.4921865 -2.6610713]
     [-2.8575273 -2.4078138 -3.4921865 -2.6610713]
     [-2.8575273 -2.4078138 -3.4921865 -2.6610713]]]]

  The mininum and maxinum are:

    -3.4921865 5.863213

  Pay more attention to the fractional part, it should have no deviation (Don't use 64-bit floating way to normalize the data)

(3)Finally, check the model output of "model_last.pth" file in the BraTS 2018 validation set. Normally, the number of the pixel per category is consistent to followings.

	  [ Brats18_CBICA_AAM_1.nii.gz ] 1: 660 | 2: 106990 | 4: 20015 WT: 127665 | TC: 20675 | ET: 20015
	  [ Brats18_CBICA_ABT_1.nii.gz ] 1: 3446 | 2: 20597 | 4: 9288 WT: 33331 | TC: 12734 | ET: 9288
	  [ Brats18_CBICA_ALA_1.nii.gz ] 1: 1519 | 2: 42794 | 4: 8813 WT: 53126 | TC: 10332 | ET: 8813
	  [ Brats18_CBICA_ALT_1.nii.gz ] 1: 49051 | 2: 48511 | 4: 50283 WT: 147845 | TC: 99334 | ET: 50283
	  [ Brats18_CBICA_ALV_1.nii.gz ] 1: 844 | 2: 62560 | 4: 1764 WT: 65168 | TC: 2608 | ET: 1764
	  [ Brats18_CBICA_ALZ_1.nii.gz ] 1: 13363 | 2: 109610 | 4: 14469 WT: 137442 | TC: 27832 | ET: 14469
	  [ Brats18_CBICA_AMF_1.nii.gz ] 1: 23958 | 2: 67703 | 4: 30071 WT: 121732 | TC: 54029 | ET: 30071
	  [ Brats18_CBICA_AMU_1.nii.gz ] 1: 15661 | 2: 121884 | 4: 40695 WT: 178240 | TC: 56356 | ET: 40695
	  [ Brats18_CBICA_ANK_1.nii.gz ] 1: 1603 | 2: 62474 | 4: 3608 WT: 67685 | TC: 5211 | ET: 3608
	  [ Brats18_CBICA_APM_1.nii.gz ] 1: 16645 | 2: 29913 | 4: 36957 WT: 83515 | TC: 53602 | ET: 36957
	  [ Brats18_CBICA_AQE_1.nii.gz ] 1: 586 | 2: 6992 | 4: 3841 WT: 11419 | TC: 4427 | ET: 3841
	  [ Brats18_CBICA_ARR_1.nii.gz ] 1: 4386 | 2: 39269 | 4: 7392 WT: 51047 | TC: 11778 | ET: 7392
	  [ Brats18_CBICA_ATW_1.nii.gz ] 1: 4549 | 2: 96631 | 4: 4594 WT: 105774 | TC: 9143 | ET: 4594
	  [ Brats18_CBICA_AUC_1.nii.gz ] 1: 22525 | 2: 7108 | 4: 12344 WT: 41977 | TC: 34869 | ET: 12344
	  [ Brats18_CBICA_AUE_1.nii.gz ] 1: 25931 | 2: 113119 | 4: 45174 WT: 184224 | TC: 71105 | ET: 45174
	  [ Brats18_CBICA_AZA_1.nii.gz ] 1: 13825 | 2: 38387 | 4: 14697 WT: 66909 | TC: 28522 | ET: 14697
	  [ Brats18_CBICA_BHF_1.nii.gz ] 1: 4959 | 2: 42572 | 4: 17834 WT: 65365 | TC: 22793 | ET: 17834
	  [ Brats18_CBICA_BHN_1.nii.gz ] 1: 13136 | 2: 39526 | 4: 23532 WT: 76194 | TC: 36668 | ET: 23532
	  [ Brats18_CBICA_BKY_1.nii.gz ] 1: 31054 | 2: 30825 | 4: 21815 WT: 83694 | TC: 52869 | ET: 21815
	  [ Brats18_CBICA_BLI_1.nii.gz ] 1: 23146 | 2: 29344 | 4: 40524 WT: 93014 | TC: 63670 | ET: 40524
	  [ Brats18_CBICA_BLK_1.nii.gz ] 1: 1456 | 2: 42838 | 4: 11973 WT: 56267 | TC: 13429 | ET: 11973
	  [ Brats18_MDA_1012_1.nii.gz ] 1: 7276 | 2: 41637 | 4: 27218 WT: 76131 | TC: 34494 | ET: 27218
	  [ Brats18_MDA_1015_1.nii.gz ] 1: 6426 | 2: 72135 | 4: 21017 WT: 99578 | TC: 27443 | ET: 21017
	  [ Brats18_MDA_1081_1.nii.gz ] 1: 14885 | 2: 10293 | 4: 46228 WT: 71406 | TC: 61113 | ET: 46228
	  [ Brats18_MDA_907_1.nii.gz ] 1: 13023 | 2: 63485 | 4: 24010 WT: 100518 | TC: 37033 | ET: 24010
	  [ Brats18_MDA_922_1.nii.gz ] 1: 5148 | 2: 105442 | 4: 22998 WT: 133588 | TC: 28146 | ET: 22998
	  [ Brats18_TCIA02_230_1.nii.gz ] 1: 19257 | 2: 87564 | 4: 72100 WT: 178921 | TC: 91357 | ET: 72100
	  [ Brats18_TCIA02_400_1.nii.gz ] 1: 582 | 2: 45964 | 4: 15007 WT: 61553 | TC: 15589 | ET: 15007
	  [ Brats18_TCIA03_216_1.nii.gz ] 1: 21854 | 2: 38098 | 4: 10024 WT: 69976 | TC: 31878 | ET: 10024
	  [ Brats18_TCIA03_288_1.nii.gz ] 1: 8051 | 2: 70868 | 4: 27834 WT: 106753 | TC: 35885 | ET: 27834
	  [ Brats18_TCIA03_313_1.nii.gz ] 1: 17708 | 2: 21056 | 4: 72393 WT: 111157 | TC: 90101 | ET: 72393
	  [ Brats18_TCIA03_604_1.nii.gz ] 1: 7786 | 2: 27640 | 4: 26942 WT: 62368 | TC: 34728 | ET: 26942
	  [ Brats18_TCIA04_212_1.nii.gz ] 1: 1293 | 2: 36010 | 4: 13666 WT: 50969 | TC: 14959 | ET: 13666
	  [ Brats18_TCIA04_253_1.nii.gz ] 1: 7308 | 2: 53424 | 4: 28659 WT: 89391 | TC: 35967 | ET: 28659
	  [ Brats18_TCIA07_600_1.nii.gz ] 1: 13848 | 2: 45231 | 4: 44677 WT: 103756 | TC: 58525 | ET: 44677
	  [ Brats18_TCIA07_601_1.nii.gz ] 1: 8550 | 2: 62828 | 4: 28702 WT: 100080 | TC: 37252 | ET: 28702
	  [ Brats18_TCIA07_602_1.nii.gz ] 1: 37989 | 2: 17851 | 4: 79179 WT: 135019 | TC: 117168 | ET: 79179
	  [ Brats18_TCIA09_248_1.nii.gz ] 1: 63126 | 2: 40845 | 4: 0 WT: 103971 | TC: 63126 | ET: 0
	  [ Brats18_TCIA10_195_1.nii.gz ] 1: 82555 | 2: 173207 | 4: 0 WT: 255762 | TC: 82555 | ET: 0
	  [ Brats18_TCIA10_311_1.nii.gz ] 1: 878 | 2: 10419 | 4: 884 WT: 12181 | TC: 1762 | ET: 884
	  [ Brats18_TCIA10_609_1.nii.gz ] 1: 56871 | 2: 33749 | 4: 0 WT: 90620 | TC: 56871 | ET: 0
	  [ Brats18_TCIA11_612_1.nii.gz ] 1: 16557 | 2: 8222 | 4: 0 WT: 24779 | TC: 16557 | ET: 0
	  [ Brats18_TCIA12_613_1.nii.gz ] 1: 1578 | 2: 556 | 4: 0 WT: 2134 | TC: 1578 | ET: 0
	  [ Brats18_TCIA13_610_1.nii.gz ] 1: 575 | 2: 95044 | 4: 12890 WT: 108509 | TC: 13465 | ET: 12890
	  [ Brats18_TCIA13_611_1.nii.gz ] 1: 44 | 2: 64075 | 4: 18998 WT: 83117 | TC: 19042 | ET: 18998
	  [ Brats18_TCIA13_617_1.nii.gz ] 1: 72325 | 2: 16624 | 4: 1132 WT: 90081 | TC: 73457 | ET: 1132
	  [ Brats18_TCIA13_636_1.nii.gz ] 1: 120225 | 2: 106942 | 4: 31470 WT: 258637 | TC: 151695 | ET: 31470
	  [ Brats18_TCIA13_638_1.nii.gz ] 1: 173655 | 2: 119288 | 4: 3050 WT: 295993 | TC: 176705 | ET: 3050
	  [ Brats18_TCIA13_646_1.nii.gz ] 1: 510 | 2: 23455 | 4: 0 WT: 23965 | TC: 510 | ET: 0
	  [ Brats18_TCIA13_652_1.nii.gz ] 1: 1912 | 2: 54942 | 4: 0 WT: 56854 | TC: 1912 | ET: 0
	  [ Brats18_UAB_3446_1.nii.gz ] 1: 1326 | 2: 103800 | 4: 12592 WT: 117718 | TC: 13918 | ET: 12592
	  [ Brats18_UAB_3448_1.nii.gz ] 1: 1156 | 2: 25911 | 4: 14569 WT: 41636 | TC: 15725 | ET: 14569
	  [ Brats18_UAB_3449_1.nii.gz ] 1: 1190 | 2: 95345 | 4: 7523 WT: 104058 | TC: 8713 | ET: 7523
	  [ Brats18_UAB_3454_1.nii.gz ] 1: 5179 | 2: 137003 | 4: 16758 WT: 158940 | TC: 21937 | ET: 16758
	  [ Brats18_UAB_3455_1.nii.gz ] 1: 5075 | 2: 78778 | 4: 11391 WT: 95244 | TC: 16466 | ET: 11391
	  [ Brats18_UAB_3456_1.nii.gz ] 1: 865 | 2: 47367 | 4: 11419 WT: 59651 | TC: 12284 | ET: 11419
	  [ Brats18_UAB_3490_1.nii.gz ] 1: 4982 | 2: 134250 | 4: 27283 WT: 166515 | TC: 32265 | ET: 27283
	  [ Brats18_UAB_3498_1.nii.gz ] 1: 3300 | 2: 85308 | 4: 16039 WT: 104647 | TC: 19339 | ET: 16039
	  [ Brats18_UAB_3499_1.nii.gz ] 1: 12223 | 2: 48541 | 4: 6028 WT: 66792 | TC: 18251 | ET: 6028
	  [ Brats18_WashU_S036_1.nii.gz ] 1: 22553 | 2: 43463 | 4: 46919 WT: 112935 | TC: 69472 | ET: 46919
	  [ Brats18_WashU_S037_1.nii.gz ] 1: 4779 | 2: 89016 | 4: 10929 WT: 104724 | TC: 15708 | ET: 10929
	  [ Brats18_WashU_S041_1.nii.gz ] 1: 8844 | 2: 85824 | 4: 18908 WT: 113576 | TC: 27752 | ET: 18908
	  [ Brats18_WashU_W033_1.nii.gz ] 1: 17041 | 2: 109986 | 4: 26219 WT: 153246 | TC: 43260 | ET: 26219
	  [ Brats18_WashU_W038_1.nii.gz ] 1: 5040 | 2: 30834 | 4: 8887 WT: 44761 | TC: 13927 | ET: 8887
	  [ Brats18_WashU_W047_1.nii.gz ] 1: 8277 | 2: 106962 | 4: 15206 WT: 130445 | TC: 23483 | ET: 15206
	  [ Brats18_WashU_W053_1.nii.gz ] 1: 3080 | 2: 29057 | 4: 12695 WT: 44832 | TC: 15775 | ET: 12695

  where "1", "2" and "4" are corresponding to label 1, 2 and 4 respectively; "WT", "TC" and "ET" are corresponding to their regions.
  
  Hoping that above informations can save your time to check the differences when making inference with our DMFNet pth file. You can raise your problem in github, I will reply them aperiodically.
