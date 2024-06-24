import os,site

now_dir = os.path.dirname(os.path.abspath(__file__))
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]

for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/ViViD.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/ViViD\n"
                    % (now_dir,now_dir)
                )
            break
        except PermissionError:
            raise PermissionError

if os.path.isfile("%s/ViViD.pth" % (site_packages_root)):
    print("!!!ViViD path was added to " + "%s/ViViD.pth" % (site_packages_root) 
    + "\n if meet No module named 'ViViD' error,please restart comfyui")


from .nodes import LoadVideo,PreViewVideo,LoadImagePath,ViViD_Node
WEB_DIRECTORY = "./web"
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadVideo": LoadVideo,
    "PreViewVideo": PreViewVideo,
    "ViViD_Node": ViViD_Node,
    "LoadImagePath": LoadImagePath
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ViViD_Node": "ViViD_Node",
    "LoadVideo": "Video Loader",
    "PreViewVideo": "PreView Video",
    "LoadImagePath": "LoadImagePath"
}