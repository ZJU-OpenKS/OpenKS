import numpy as np
import torch
import os



# TODO
def save_bbox_heatmap(bboxes, heatmap, save_base = os.getcwd()+'/heatmap_result', save_name='', kth_input=None): # save_name: text
    #print("bboxes", bboxes.shape)
    #print("heatmap", heatmap.shape)
    color = [1,1,1]
    #color = color.cpu().numpy()
    save_base = os.path.join(save_base, save_name)
    print(save_base, flush=True)
    if not os.path.exists(save_base):
        os.makedirs(save_base)
    if kth_input is not None:
        for idx in range(heatmap.shape[0]):  # idx: object
            kth = kth_input
            kth = int(kth)
            obj_name = str(idx) + '_' + str(int(kth)) + '.obj'
            real_save_path = os.path.join(save_base, obj_name)
            file_out = open(real_save_path, 'w')
            norm = heatmap[idx][kth].max()
            # print(real_save_path, '<< save path; norm=', norm)
            for _, point in enumerate(bboxes):
                p = (heatmap[idx][kth][_].cpu()*5).numpy()
                p = min(1, p)
                point = point[[0,4,1,5,3,7,2,6], :]
                # print(p, '<< heatmap')
                # pc, sz = point[:3], point[3:] / 2
                # off = [[1, 1, 1, 1, -1, -1, -1, -1],
                #        [1, 1, -1, -1, 1, 1, -1, -1],
                #        [1, -1, 1, -1, 1, -1, 1, -1]]
                # print("pc", pc.shape)
                # print("sz", sz.shape)
                for i in range(8):
                    # print("off[0][i]", off[0][i])
                    # print("pc[0]+sz[0]*off[0][i]", pc[0] + sz[0] * off[0][i])
                    # print("pc[1]+sz[1]*off[1][i]", pc[1]+sz[1]*off[1][i])
                    # print("pc[2]+sz[2]*off[2][i]", pc[2]+sz[2]*off[2][i])
                    # print("color[0]", color[0], p)
                    # print("color[0]*(1-p)+p", color[0] * (1 - p) + p)
                    # print("color[1]*(1-p)", color[1]*(1-p))
                    # print("color[2]*(1-p)", color[2]*(1-p))
                    # print('v %f %f %f %f %f %f' % (
                    # pc[0] + sz[0] * off[0][i], pc[1] + sz[1] * off[1][i], pc[2] + sz[2] * off[2][i],
                    # color[0] * (1 - p) + p, color[1] * (1 - p), color[2] * (1 - p)), file=file_out)
                    print('v %f %f %f %f %f %f' % (point[i][0], point[i][1], point[i][2],
                        color[0] * (1 - p) + p, color[1] * (1 - p), color[2] * (1 - p)), file=file_out)
                bs = 8 * _
                # TODO 2 face
                print('f %d %d %d %d' % (1 + bs, 2 + bs, 4 + bs, 3 + bs), file=file_out)
                print('f %d %d %d %d' % (5 + bs, 6 + bs, 8 + bs, 7 + bs), file=file_out)
                print('f %d %d %d %d' % (1 + bs, 2 + bs, 6 + bs, 5 + bs), file=file_out)
                print('f %d %d %d %d' % (3 + bs, 4 + bs, 8 + bs, 7 + bs), file=file_out)

                print('f %d %d %d %d' % (1 + bs, 3 + bs, 7 + bs, 5 + bs), file=file_out)
                print('f %d %d %d %d' % (2 + bs, 4 + bs, 8 + bs, 6 + bs), file=file_out)

                print('l %d %d' % (1 + bs, 2 + bs), file=file_out)
                print('l %d %d' % (2 + bs, 4 + bs), file=file_out)
                print('l %d %d' % (4 + bs, 3 + bs), file=file_out)
                print('l %d %d' % (3 + bs, 1 + bs), file=file_out)

                print('l %d %d' % (5 + bs, 6 + bs), file=file_out)
                print('l %d %d' % (6 + bs, 8 + bs), file=file_out)
                print('l %d %d' % (8 + bs, 7 + bs), file=file_out)
                print('l %d %d' % (7 + bs, 5 + bs), file=file_out)

                print('l %d %d' % (2 + bs, 6 + bs), file=file_out)
                print('l %d %d' % (5 + bs, 1 + bs), file=file_out)
                print('l %d %d' % (4 + bs, 8 + bs), file=file_out)
                print('l %d %d' % (7 + bs, 3 + bs), file=file_out)
                # print('l 1 2', file=file_out)
                # print('l 2 4', file=file_out)
                # print('l 4 3', file=file_out)
                # print('l 3 1', file=file_out)
                # print('l 5 6', file=file_out)
                # print('l 6 8', file=file_out)
                # print('l 8 7', file=file_out)
                # print('l 7 5', file=file_out)
                # print('l 2 6', file=file_out)
                # print('l 5 1', file=file_out)
                # print('l 4 8', file=file_out)
                # print('l 7 3', file=file_out)
            file_out.close()
    else:
        raise NotImplementedError()
