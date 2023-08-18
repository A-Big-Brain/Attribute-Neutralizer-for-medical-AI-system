import run
import support_args as spa

ar = spa.parse()

da_li = ['ChestX-ray14', 'CheXpert', 'MIMIC']
fo_li = [['ChestX-ray14_gender_age_3_5_100.0_10.0_0.0_JUq3', 'ChestX-ray14_gender_age_3_50_100.0_10.0_0.0_VGct'],
         ['CheXpert_gender_age_3_2_100.0_10.0_0.0_bY9z'],
         ['MIMIC_gender_3_2_100.0_10.0_0.0_sP4W']]

con_li = [[da_li[i], fo_li[i][j]] for i in range(3) for j in range(len(fo_li[i]))]

run.run_model(da_tr=con_li[ar.conf_num][0], folder=con_li[ar.conf_num][1], model_num=ar.model_num, bat_num=ar.bat_num, ep=ar.epoch, tr_it=ar.tr_it, te_it = ar.te_it, pr_it=ar.pr_it)