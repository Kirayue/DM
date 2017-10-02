import requests
import re
import json
import sys
import os
def readFile(filename):
    with open(filename, 'r') as f:
        rawData = f.read()
        rawData = rawData.split('\n')
        del rawData[-1]
    return rawData

if __name__ == '__main__':
    pat1 = '<meta property="og:image" content="(.+)"  data-dynamic="true">'
    pat2 = 'modelExport:(.+),\s+auth'
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    image_link = []
    missing_data_index = []
    unknowissue_data_index = []
    metadata = {}
    rawData = readFile('./t1_train_image_link.txt')
    for index, link in enumerate(rawData):
        if index < start:
            continue
        if index >= end:
            break
        print("Link url: {}".format(link))
        res = requests.request("GET", link)
        #match = re.search(pat, res.text)
        if res.status_code == 404:
            missing_data_index.append(index)
            continue
        match1 = re.match(pat1, res.text)
        print(pat1)
        print(res.text)
        print(match1)
        if not match1:
            print(match1)
            exit()
            image_link.append(match1.group(1))
            an_image_link = match1.group(1)
            print('{} : {}'.format(len(image_link), match1.group(1)))
            os.system('wget -O ./part/image/{}.png {}'.format(index, an_image_link))
            match2 = re.search(pat2, res.text)
            full_json = json.loads(match2.group(1))
            print(full_json['photo-head-meta-models'][0]['id'])
            metadata[index] = {
                    'title': full_json['photo-models'][0]['title'] if 'title' in full_json['photo-models'][0] else "",
                    'description': full_json['photo-models'][0]['description'] if 'description' in full_json['photo-models'][0] else "",
                    'canComment': full_json['photo-models'][0]['canComment'] if 'canComment' in full_json['photo-models'][0] else False,
                    'canPublicComment': full_json['photo-models'][0]['canPublicComment'] if 'canPublicComment' in full_json['photo-models'][0] else False,
                    'faveCount': full_json['photo-models'][0]['engagement']['faveCount'] if 'faveCount' in full_json['photo-models'][0]['engagement'] else 0,
                    'commentCount': full_json['photo-models'][0]['engagement']['commentCount'] if 'commentCount' in full_json['photo-models'][0]['engagement'] else 0,
                    'viewCount': full_json['photo-models'][0]['engagement']['viewCount'] if 'viewCount' in full_json['photo-models'][0]['engagement'] else 0,
                    'isHD': full_json['photo-models'][0]['isHD'] if 'isHD' in full_json['photo-models'][0] else False,
                    'datePosted': full_json['photo-stats-models'][0]['datePosted'],
                    'keyword': full_json['photo-head-meta-models'][0]['keywords'].split(', ') if 'keywords' in full_json['photo-head-meta-models'][0] else [],
                    'num_groups': len(full_json['photo-head-meta-models'][0]['flickr_photos:groups']) if 'flickr_photos:groups' in full_json['photo-head-meta-models'][0] else 0
                    }
            #print(full_json['photo-models'][0]['isHD'])
        else:
            unknowissue_data_index.append(index)
            print('Unknow issue')

    with open('./part/image_list_' + sys.argv[3], 'w') as f:
        f.write('\n'.join(image_link))
    with open('./part/missing_list_' + sys.argv[3], 'w') as f:
        f.write('\n'.join([str(i) for i in missing_data_index]))
    with open('./part/unknow_list_' + sys.argv[3], 'w') as f:
        f.write('\n'.join([str(i) for i in unknowissue_data_index]))
    with open('./part/metadata_' + sys.argv[3], 'w') as f:
        json.dump(metadata, f)
