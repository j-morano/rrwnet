import skimage.io as io
import imageio
import numpy as np



def create_av3_groundtruths(fov_mask_name, in_img_name, out_img_name, out_mask_name):
    gt = (io.imread(in_img_name) / 255)
    fov_mask = (io.imread(fov_mask_name) / 255)
    # f.plot_image(gt)

    a = gt[:, :, 0]  # Arteries + Unknown
    c = gt[:, :, 1]  # Crossings (both Artery and Vein) + Unknown
    v = gt[:, :, 2]  # Veins + Unknown

    unk_f = a * c * v
    a_u = a - unk_f
    c_u = c - unk_f
    v_u = v - unk_f

    a_f = np.logical_or(a_u, c_u).astype('float')
    v_f = np.logical_or(v_u, c_u).astype('float')
    vt_f = np.logical_or(a_u, np.logical_or(v_u, c_u)).astype('float')

    vt_f += unk_f

    gt = np.stack((a_f, v_f, vt_f), axis=2)

    mask = a * c * v
    mask = 1 - mask
    mask = mask * fov_mask

    imageio.imwrite(out_img_name, gt)
    imageio.imwrite(out_mask_name, mask)
