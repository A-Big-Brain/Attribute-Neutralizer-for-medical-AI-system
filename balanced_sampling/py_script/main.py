import run
import support_args as spa

ar = spa.parse()

da_li = ['ChestX-ray14', 'CheXpert', 'MIMIC']


run.run_model(da_tr=da_li[ar.da_num], att_str=ar.att_str, model_num=ar.model_num,
              bat_num=ar.bat_num, ep=ar.epoch, tr_it=ar.tr_it, te_it = ar.te_it, pr_it=ar.pr_it)