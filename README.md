![](tffs.png?raw=true)

# TFFS
> TensorFlow Filesystem - Access Tensors Differently

A funny way to access your tensorflow model's tensors.

Use this project to map your model into a filesystem. Then, access your tensors as if they were files, using your favorite UNIX commands.

`tffs` is implemented using Filesystem in Userspace (FUSE). It requires `tensorflow` and [`fusepy`](https://github.com/fusepy/fusepy) to be installed.

To learn more, read the accompanying [blog post](http://anotherdatum.com/tffs.html).


## Usage
1. Create a model - out of the scope of this project :)


2. Mount your model so it'll be accessible through the filesystem:
	```bash
	python tffs.py --model PATH_TO_MODEL --mount MOUNT_POINT
	```
	PATH_TO_MODEL is either a directory containing a .meta file, or the .meta file itself.

	If there's also a file containing the weights with the same name as the .meta file (without the .meta extension), it'll be loaded as well.


3. Reap the fruits. Assuming MOUNT_POINT is ~/tf:

|                 Command                |                          Description                          |
|----------------------------------------|---------------------------------------------------------------|
| `find ~/tf`                            | list all scopes and tensors                                   |
| `find ~/tf -type f`                    | list all tensors                                              |
| `xattr -l ~/tf/.../tensor`             | get attributes of a tensor                                    |
| `cat ~/tf/.../tensor`                  | print the value found in a tensor                             |
| `~/tf/bin/inputs -d 3 ~/tf/.../tensor` | print the inputs to a tensor, recursively                     |
| `~/tf/bin/outputs --no-fs .../tensor`  | print the outputs to a tensor, without using the mount prefix |

