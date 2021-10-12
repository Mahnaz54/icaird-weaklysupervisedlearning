# This script is the Docker entry point, it is run when the container
# starts up and makes changes to the permissions inside the container.
#
# This script resets the ubuntu users ids to be those of the
# user that owns the notebooks folder, allowing it to write
# to that folder in a way that doesn't mess up the file ownership
# in the host os.  This has to be done as root but then we use
# gosu to start a session as ubuntu.

# sort out the user ids
old_uid=$(id -u ubuntu)
new_uid=$(stat -c %u ./notebooks)
usermod -u $new_uid ubuntu &
#kill $! # nasty hack - usermod is not exiting for some reason
find ./ -user $old_uid -exec chown -h $new_uid {} \;

# sort out the group ids
old_gid=$(id -g ubuntu)
new_gid=$(stat -c %g ./notebooks)
groupmod -g $new_gid ubuntu
find ./ -group $old_gid -exec chgrp -h $new_gid {} \;
gosu ubuntu bash
