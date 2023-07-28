# from nuscenes.nuscenes import NuScenes
# nusc = NuScenes(version='v1.0-mini', dataroot='D:/nuscenes_data/v1.0-mini', verbose=True)
# my_sample = nusc.sample[0]
# print(my_sample['token'])
# scene=nusc.scene[0]
# first_sample_token = scene['first_sample_token']
# sample_data_token = nusc.get('sample',first_sample_token)['data']['LIDAR_TOP']
# cur_sd=nusc.get('sample_data',sample_data_token)
#
# cur_sample=nusc.get('sample',cur_sd['sample_token'])
# print(cur_sample['token'])
#
# print(my_sample['scene_token'],scene['token'])
import simplejson as json
# with open ('D:/vis_occ/mini-data/annotations.json','r') as f:
with open ('D:/annotations.json','r') as f:
    data=json.load(f)
    print(data['scene_infos']['scene-0061']['ca9a282c9e77460f8360f564131a8af5']['gt_path'])
    # print(data.keys())