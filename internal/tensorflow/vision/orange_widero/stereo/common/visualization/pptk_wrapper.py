import numpy as np
import matplotlib.colors as mcolors
import termcolor
import colorsys


def project_points_to_image_plane(view, points):
    f, RT = view['focal'], view['RT_view_to_main']
    R, T = RT[0:3,0:3], RT[0:3,-1]
    rotated_points = R.T.dot((points - T).T)
    X, Y, Z = rotated_points
    ox, oy = view['origin']
    projected_points = np.array([f * X / Z + ox, f * Y / Z + oy]).T
    projected_points[Z < 0] = [-1000, -1000]
    return projected_points.T

def are_inside(view, points):
    x, y = project_points_to_image_plane(view, points)
    x, y = np.round(np.array(x)).astype(int), np.round(np.array(y)).astype(int)
    h, w = view['image'].shape
    inside = (x < w) * (x >= 0) * (y < h) * (y >= 0)
    inside[inside==True] = view['image'][y[inside==True], x[inside==True]] != 0
    return inside

def mask_color_pts(mask, color_pts, colored=False, color='m'):
    import matplotlib.colors as mcolors
    color = mcolors.to_rgba(color)
    if color_pts.ndim == 1 or color_pts.shape[-1] == 1:
        rgb_mask = np.c_[(mask,)*4] * color
    rgb_grayscale = np.c_[(color_pts,)*3]
    # add alpha
    rgba_grayscale = np.c_[rgb_grayscale, np.ones_like(rgb_grayscale[..., 0])]
    if colored:
        rgba_grayscale[mask] = rgb_mask[mask]
    else:
        # masked points will be transparent
        rgba_grayscale[mask, 3] = 0
    return rgba_grayscale


class ViewWrapper(object):
    """
    Wraps a pptk3d viewer object

    Init params
    ==========

    pptk3d_viewer: A ptk3d viewer object
    grayscale: The grayscale of the viewer's point cloud
    """
    def __init__(self, pptk3d_viewer, grayscale):
        self.viewer = pptk3d_viewer
        self.grayscale = grayscale
        self.attributes = []
        self.attributes_drescriptors = []
        self.views = None
        self.points_to_views = None
    
    @staticmethod
    def _get_views(clip_name, views_names, view_gen):
        #TODO: generate this data in a more reasonable way (and get only the data which isn't gi depended)
        if view_gen is None:
            from stereo.data.view_generator.view_generator import ViewGenerator
            view_gen = ViewGenerator(clip_name, views_names, mode='pred', no_labels=True)
        for i in np.arange(10):
            try:
                first_gi = view_gen.sub_clip.gfi_to_grab['main'][view_gen.sub_clip.first_gfi()+i][0]
                views = view_gen.get_gi_views(first_gi)
                return views
            except:
                pass
        raise Exception("Couldn't get views")
    
    def _update_attributes(self):
        self.viewer.attributes(self.grayscale, *self.attributes)

    def print_layers(self):
        print("Current layers:")
        for descriptor in self.attributes_drescriptors:
            white_list, black_list, name, color = descriptor
            print(termcolor.colored(("Layer - \"%s\", Points that -\n\tAppear in all the views - %s\n\t Do not appear in any of the views - %s" %
                                     (name, white_list, black_list)), color))
        
    def clear_views_layers(self):
        """
        Clear all colored layers from viewer
        """
        self.attributes = []
        self.attributes_drescriptors = []
        self._update_attributes()
    
    def register_views(self, clip_name, views_names, pcd, view_gen=None):
        """
        Initialize views dictinoaries related to the point cloud the viewer shows
        and calculate which point is viewable from which view
        Params
        ==========

        clip_name (str):     The clip from which the point cloud was generated
        views_names (list):  The names of the views to register
        pcd (array):         The point cloud of the view
        view_gen(opptional): A ViewGenerator object           
        """
        views = ViewWrapper._get_views(clip_name, views_names, view_gen)
        points_to_views = np.zeros((len(pcd),len(views)), dtype=np.bool)
        for i, view in enumerate(views):
            points_to_views[:, i] = are_inside(views[view], pcd)
        self.views = views
        self.points_to_views = points_to_views
    
    def add_views_layer(self, white_list, black_list, name='', color='red', verbose=False):
        """
        Add to the viewer colored layer of points which appear in some views and 
        do not appear in other views
        ==========

        white_list (list):      List of views where the points appear (must appear in all views)
        black_list (list):      List of views where the points do not appear (may not appear in any of the views)
        name  (str, optional):  The name of the layer
        color (str, optional):  The color of the layer. Available colors: 
                                red, blue, yellow, green, grey, cyan, magenta (purple), white
        verbose (bool):         Print current layers
        """
        if not self.views:
            print("Views aren't registered yet. call register_views first")
            return
        if color not in termcolor.COLORS:
            print("Color not in availale colormap. The full colormap is:\n%s" % termcolor.COLORS.keys())
            return
        colors = np.c_[self.grayscale, self.grayscale, self.grayscale]
        white_indices = [self.views.keys().index(vi) for vi in white_list]
        black_indices = [self.views.keys().index(vi) for vi in black_list]
        all_indices = white_indices + black_indices
        indeices_values = [True] * len(white_indices) + [False] * len(black_indices)
        points_in_intersection = (self.points_to_views[:, all_indices] == indeices_values).all(axis=1)
        n = len(self.grayscale[points_in_intersection])
        if n == 0:
            print("No points in this intersection")
            return
        h, s, _ = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
        hsv = np.c_[np.ones(n)*h, np.ones(n)*s, self.grayscale[points_in_intersection]]
        rgb = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in hsv]
        colors[points_in_intersection] = rgb
        self.attributes.append(colors)
        if name == '':
            name = "layer_%s" % len(self.attributes)
        self.attributes_drescriptors.append([white_list, black_list, name, color])
        self._update_attributes()
        if verbose:
            self.print_layers()
        self.viewer.set(curr_attribute_id = len(self.attributes))

    def add_non_stereo_layer(self, section_name, color='red', verbose=False):
        """
        Add to the viewer colored layer of points which appear in some section only in the center camera
        ==========
        section_name (str): The main camera of the layer
        color (str, optional): The color of the layer. Available colors: 
                               red, blue, yellow, green, grey, cyan, magenta (purple), white
        verbose (bool):         Print current layers
        """
        white_list = [v for v in self.views if v.endswith(section_name) and v.startswith(section_name)]
        black_list = [v for v in self.views if v.endswith(section_name) and not v.startswith(section_name)]
        name = 'only_%s_camera' % section_name
        self.add_views_layer(white_list, black_list, name, color, verbose)

    def join_layers(self, layers_names, color='red', verbose=False):
        """
        Add to the viewer colored layer of the union of two existing layers
        ==========
        layers_names (list): List of layers, by their names
        color (str, optional): The color of the new layer. Available colors: 
                               red, blue, yellow, green, grey, cyan, magenta (purple), white
        verbose (bool):         Print current layers
        """
        layers_indices = {descriptor[2]:index for index, descriptor in enumerate(self.attributes_drescriptors)}
        joint_attributes = np.c_[self.grayscale, self.grayscale, self.grayscale]
        for l in layers_names:
            if l in layers_indices:
                index = layers_indices[l]
                attr = self.attributes[index]
                colored_indeices = np.min(attr,axis=1) != np.max(attr, axis=1)
                n = len(self.grayscale[colored_indeices])
                h, s, _ = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
                hsv = np.c_[np.ones(n)*h, np.ones(n)*s, self.grayscale[colored_indeices]]
                rgb = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in hsv]
                joint_attributes[colored_indeices] = rgb
            else:
                print("layer %s isn't part of the view" % l)
                return
        self.attributes.append(joint_attributes)
        name = " U ".join(layers_names)
        self.attributes_drescriptors.append([['?'], ['?'], name, color])
        self._update_attributes()
        self.viewer.set(curr_attribute_id = len(self.attributes))

    def show_confidence(self, error_pts, threshold=0.04, mask_color='m'):
        self.attributes.append(error_pts)
        mask = error_pts > threshold
        self.attributes.extend([mask_color_pts(mask, self.grayscale, colored=True, color=mask_color),
                                mask_color_pts(mask, self.grayscale, colored=False)])
        self.viewer.attributes(np.c_[(self.grayscale,) * 3], *self.attributes)
        self.viewer.color_map('jet', scale=[0, 0.5])
        self.viewer.set(curr_attribute_id=len(self.attributes))

