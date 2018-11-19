import matplotlib.pyplot as plt
import os

### dpp 10frame val##
#cider_to_draw = [[0.757356298222],[0.788177403787],[0.817152407125],[0.784629041391],[0.847209608686],[0.88657611764], [0.923346784982],[0.866131214785],[0.802423967783],[0.759015279935],[0.781671319222],
#[0.782425091441],[0.765683658201],[0.764946629],[0.738107157723],[0.83096770505],[0.800713379903],[0.822797119833],[0.780272291788]]
#####
######### dpp 25frame test####
#cider_to_draw = [[0.602477593163],[0.703465492652],[0.710464812487],[0.688701980191], [0.719196089706], [0.707658532871], [0.718357897286], [0.688883591056], [0.718275964565], [0.702449222895], #[0.693678007986], [0.70243349874], [0.728516407595], [0.716709397516], [0.703625822559], [0.714006108033],[0.703571633384], [0.704417288813], [0.701158331816],[0.703612171363] ]
#################
cider_to_draw = []
filename = './dpp_results/1frame_dup/train_dup3frame_adam_on_test.txt'
with open(filename,'r') as f:
  for line in f:
    if 'CIDEr' in line:
      cider_to_draw.append([line.split(':')[-1]])
    

plt_save_dir = "./dpp_results/1frame_dup"
plt_save_img_name = 'dup3frame_adam_cider_test.png'
plt.plot(range(len(cider_to_draw)), cider_to_draw, color='g')
plt.grid(True)
plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))
