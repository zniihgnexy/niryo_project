import mujoco_py

def save_mjcf():
    model = mujoco_py.load_model_from_path("mjmodel.xml")
    sim = mujoco_py.MjSim(model)
    # Run your simulation or do manipulations here
    mjcf_content = mujoco_py.functions.mj_saveLastXML(sim.model, sim.data, "mjmodel_mjcf.mjcf", None)
    with open("path_to_save_new_file.mjcf", "w") as file:
        file.write(mjcf_content)

save_mjcf()
