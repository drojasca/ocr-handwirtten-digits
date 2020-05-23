tmux new -s Detector -d
tmux rename-window -t Detector roscore
tmux send-keys -t Detector 'cd catkin_ws/' C-m
tmux send-keys -t Detector 'roscore' C-m

tmux new-window -t Detector
tmux rename-window -t Detector detection
tmux send-keys -t Detector 'cd catkin_ws/' C-m
tmux send-keys -t Detector 'source devel/setup.bash' C-m
tmux send-keys -t Detector 'rosrun visualize Detector' C-m
tmux split-pane -v -t Detector

tmux send-keys -t Detector 'cd catkin_ws/' C-m
tmux send-keys -t Detector 'source devel/setup.bash' C-m
tmux send-keys -t Detector 'rosrun visualize rqt_image_view ' C-m
tmux split-pane -v -t Detector

tmux send;keys -t Detector '<C-b>R Image Node' C-m
tmux send-keys -t Detector 'cd catkin_ws/' C-m
tmux send-keys -t Detector 'source devel/setup.bash' C-m
tmux send-keys -t Detector 'rosrun visualize Image' C-m


tmux attach -t Detector
