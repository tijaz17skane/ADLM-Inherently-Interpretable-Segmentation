import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

from tqdm import tqdm

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop


# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel,  # pytorch network with prototype_vectors
                    preprocess_input_function=None,  # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved here
                    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,  # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):
    if hasattr(prototype_network_parallel, 'module'):
        prototype_network_parallel = prototype_network_parallel.module

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_network_parallel.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-' + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.num_classes

    log(f'Updating prototypes...')
    for push_iter, (search_batch_input, search_y) in tqdm(enumerate(dataloader),
                                                          desc='updating prototypes', total=len(dataloader)):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir,
                             proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end - start))


# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist,  # this will be updated
                               global_min_fmap_patches,  # this will be updated
                               proto_rf_boxes,  # this will be updated
                               proto_bound_boxes,  # this will be updated
                               search_y=None,  # required if class_specific == True
                               num_classes=None,  # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):
    if hasattr(prototype_network_parallel, 'module'):
        prototype_network_parallel = prototype_network_parallel.module

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    class_to_img_index_dict = {key: [] for key in range(num_classes)}

    # img_y is the image's integer label
    for img_index, img_y in enumerate(search_y):
        for proto_i in range(img_y.shape[0]):
            for proto_j in range(img_y.shape[1]):
                for cls_i, cls_prob in enumerate(img_y[proto_i, proto_j]):
                    if cls_prob > 0:
                        class_to_img_index_dict[cls_i].append((img_index, proto_i, proto_j))

    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):
        # target_class is the class of the class_specific prototype
        target_class = torch.argmax(prototype_network_parallel.prototype_class_identity[j]).item()
        # if there is not images of the target_class from this batch
        # we go on to the next prototype
        if len(class_to_img_index_dict[target_class]) == 0:
            continue

        # proto_dist_.shape = (b_size, n_prototypes, patches_rows, patches_cols)
        all_dist = np.asarray([proto_dist_[img_index, j, proto_i, proto_j]
                               for img_index, proto_i, proto_j in class_to_img_index_dict[target_class]])

        batch_argmin_proto_dist = np.argmin(all_dist)
        batch_min_proto_dist = all_dist[batch_argmin_proto_dist]

        if batch_min_proto_dist < global_min_proto_dist[j]:
            '''
            change the argmin index from the index among
            images of the target class to the index in the entire search
            batch
            '''
            batch_argmin_proto_dist = class_to_img_index_dict[target_class][batch_argmin_proto_dist]
            img_index_in_batch, proto_i, proto_j = batch_argmin_proto_dist

            # retrieve the corresponding feature map patch
            fmap_height_start_index = proto_i * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = proto_j * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            # ProtoL.shape = (1, 512, 8, 16)

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                     :,
                                     fmap_height_start_index:fmap_height_end_index,
                                     fmap_width_start_index:fmap_width_end_index]

            # batch_min_fmap_patch_j.shape = (512, 1, 1)

            global_min_proto_dist[j] = batch_min_proto_dist
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = prototype_network_parallel.proto_layer_rf_info
            # rf_prototype_j = compute_rf_prototype((search_batch.shape[2], search_batch.shape[3]),
            # batch_argmin_proto_dist, protoL_rf_info)

            # TODO patch size - not hardcoded
            patch_size = 64
            rf_start_h_index, rf_end_h_index = batch_argmin_proto_dist[1] * patch_size, \
                                               (batch_argmin_proto_dist[1] + 1) * patch_size
            rf_start_w_index, rf_end_w_index = batch_argmin_proto_dist[2] * patch_size, \
                                               (batch_argmin_proto_dist[2] + 1) * patch_size

            rf_prototype_j = [img_index_in_batch, rf_start_h_index, rf_end_h_index, rf_start_w_index, rf_end_w_index]

            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_height = original_img_j.shape[0]
            original_img_width = original_img_j.shape[1]

            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                       rf_prototype_j[3]:rf_prototype_j[4], :]

            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = target_class

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if prototype_network_parallel.prototype_activation_function == 'log':
                proto_act_img_j = np.log(
                    (proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.epsilon))
            elif prototype_network_parallel.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_width, original_img_height),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                          proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = target_class

            '''
            proto_rf_boxes and proto_bound_boxes column:
            0: image index in the entire dataset
            1: height start index
            2: height end index
            3: width start index
            4: width end index
            5: (optional) class identity
            '''
            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png

                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + f'_{j}-original.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    plt.figure(figsize=(20.48, 10.24))  # for 100 DPI
                    plt.imshow(original_img_j)
                    plt.plot([rf_start_w_index, rf_start_w_index], [rf_start_h_index, rf_end_h_index],
                             [rf_end_w_index, rf_end_w_index], [rf_start_h_index, rf_end_h_index],
                             [rf_start_w_index, rf_end_w_index], [rf_start_h_index, rf_start_h_index],
                             [rf_start_w_index, rf_end_w_index], [rf_end_h_index, rf_end_h_index],
                             linewidth=2, color='red')
                    plt.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                        hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.savefig(os.path.join(dir_for_saving_prototypes,
                                             prototype_img_filename_prefix + f'_{j}-original_with_box.png'))
                    plt.close()

                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]

                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + f'_{j}-original_with_self_act.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    plt.figure(figsize=(20.48, 10.24))  # for 100 DPI
                    plt.imshow(overlayed_original_img_j)
                    plt.plot([rf_start_w_index, rf_start_w_index], [rf_start_h_index, rf_end_h_index],
                             [rf_end_w_index, rf_end_w_index], [rf_start_h_index, rf_end_h_index],
                             [rf_start_w_index, rf_end_w_index], [rf_start_h_index, rf_start_h_index],
                             [rf_start_w_index, rf_end_w_index], [rf_end_h_index, rf_end_h_index],
                             linewidth=2, color='red')
                    plt.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                        hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.savefig(os.path.join(dir_for_saving_prototypes,
                                             prototype_img_filename_prefix + f'_{j}-original_with_self_act_and_box.png'))
                    plt.close()

                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + f'_{j}-receptive_field.png'),
                               rf_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                         rf_prototype_j[3]:rf_prototype_j[4]]
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + f'_{j}-receptive_field_with_self_act.png'),
                               overlayed_rf_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + f'_{j}.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)

    del class_to_img_index_dict
