import requests
import re
import json

def readFile(filename):
    with open(filename, 'r') as f:
        rawData = f.read()
        rawData = rawData.split('\n')
        del rawData[-1]
    return rawData

if __name__ == '__main__':
    pat = '<meta property="og:image" content="(.+)"  data-dynamic="true">'
    image_link = []
    missing_data_index = []
    login_data_index = []
    multiple_data_index = []
    unknowissue_data_index = []
    metadata = {}
    rawData = readFile('./t1_train_image_link.txt')
    for index, link in enumerate(rawData):
        print("Link url: {}".format(link))
        res = requests.request("GET", link)
        #match = re.search(pat, res.text)
        if res.status_code == 404:
            missing_data_index.append(index)
            continue
        test = 0
        for match in re.finditer(pat, res.text):
            image_link.append(match.group(1))
            test += 1
            print('{} : {}'.format(len(image_link),match.group(1)))
        if test > 1:
            multiple_data_index.append(index)
            print("Multiple match issue")
            print("test: {}".format(test))
            print("index: {}".format(index))
        elif re.search('Yahoo - 登入', res.text):
            login_data_index.append(index)
            print('Login issue')
        else:
            unknowissue_data_index.append(index)
            print('Unknow issue')
        break

    with open('./part/image_list_' + sys.argv[2], 'w') as f:
        f.write('\n'.join(image_link))
    with open('./part/missing_list_' + sys.argv[2], 'w') as f:
        f.write('\n'.join(missing_data_index))
    with open('./part/login_list_' + sys.argv[2], 'w') as f:
        f.write('\n'.join(login_data_index))
    with open('./part/multiple_list_' + sys.argv[2], 'w') as f:
        f.write('\n'.join(multiple_data_index))
    with open('./part/unknow_list_' + sys.argv[2], 'w') as f:
        f.write('\n'.join(unknowissue_data_index))
    with open('./part/metadata_' + sys.argv[2], 'w') as f:
        json.dump(metadata, f)
