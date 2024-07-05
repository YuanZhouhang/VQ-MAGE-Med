import os
import pandas as pd
from tqdm import trange

output_dir = './Result/MedAug'
dataset_list = ['ham10000', 'ODIR-5k']
dataset = dataset_list[0]

if dataset == 'ham10000':
    # ham10000
    root = 'INSERT_HAM10K_ROOT'
    data = pd.read_csv(root+'/ISIC2018_Task3_Test_GroundTruth.csv')
    category = data['dx'].unique()
    print(category)

    for i in category:
        os.makedirs(os.path.join(output_dir, 'ham10000', 'test', i), exist_ok=True)

    # output image number of each category
    for i in category:
        print(i, len(data[data['dx'] == i]))

    for i in range(len(data)):
        image_name = data['image_id'][i] + '.jpg'
        image_label = data['dx'][i]
        # image copy to new dir
        os.system(f'cp {root}/ISIC2018_Task3_Test_Images/' + image_name + ' ' + os.path.join(output_dir, 'ham10000', 'test' , image_label, image_name))

elif dataset == 'ODIR-5k':
    # ODIR-5k
    root = 'INSERT_ODIR5K_ROOT'
    keyword_list = ['normal', 'proliferative retinopathy', 'glaucoma', 'cataract', 'age-related macular degeneration', 'hypertensive', 'myopia']
    category_list = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

    def odir_move_image(data, set_name, set_rename):
        image_number = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in trange(len(data)):
            left_image_name = data['Left-Fundus'][i]
            right_image_name = data['Right-Fundus'][i]
            left_image_keyword = data['Left-Diagnostic Keywords'][i]
            right_image_keyword = data['Right-Diagnostic Keywords'][i]
            left_other = False
            right_other = False
            os.makedirs(os.path.join(output_dir, 'ODIR', set_rename, 'Other'), exist_ok=True)
            for k in range(len(keyword_list)):
                os.makedirs(os.path.join(output_dir, 'ODIR', set_rename, category_list[k]), exist_ok=True)
                if keyword_list[k] in left_image_keyword:
                    os.system(
                        f'cp {root}' + set_name + '/Images/' + left_image_name + ' ' + os.path.join(
                            output_dir, 'ODIR', set_rename, category_list[k], left_image_name))
                    left_other = True
                    image_number[k] += 1
                if keyword_list[k] in right_image_keyword:
                    os.system(
                        f'cp {root}' + set_name + '/Images/' + right_image_name + ' ' + os.path.join(
                            output_dir, 'ODIR', set_rename, category_list[k], right_image_name))
                    right_other = True
                    image_number[k] += 1
            if left_other == False:
                os.system(
                    f'cp {root}' + set_name + '/Images/' + left_image_name + ' ' + os.path.join(
                        output_dir, 'ODIR', set_rename, 'Other', left_image_name))
                image_number[7] += 1
            if right_other == False:
                os.system(
                    f'cp {root}' + set_name + '/Images/' + right_image_name + ' ' + os.path.join(
                        output_dir, 'ODIR', set_rename, 'Other', right_image_name))
                image_number[7] += 1

        print(set_name, image_number)

    # training set
    data = pd.read_excel(f'{root}/Training_Set/Annotation/training annotation (English).xlsx')
    odir_move_image(data, 'Training_Set', 'train')

    # valid set
    data = pd.read_excel(f'{root}/Off-site_Test_Set/Annotation/off-site test annotation (English).xlsx')
    odir_move_image(data, 'Off-site_Test_Set', 'test')

    # test set
    data = pd.read_excel(f'{root}/On-site_Test_Set/Annotation/on-site test annotation (English).xlsx')
    odir_move_image(data, 'On-site_Test_Set', 'test')
