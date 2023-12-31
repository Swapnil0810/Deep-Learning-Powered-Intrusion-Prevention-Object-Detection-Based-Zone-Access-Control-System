width, height = 640, 480
from ultralytics import YOLO
from service import counting_in_out_person_test
import traceback
from sort import *#changes
from service_skeleton.http_server import *#changes
from skeleton_updater import check_for_lib_updates  #changes
# DJANGO_API_ENDPOINT = 'http://192.168.1.237:8000'
# ALERT_SEND_URL = f"{DJANGO_API_ENDPOINT}/service_app/create_alert/"
check_for_lib_updates(DJANGO_API_ENDPOINT) #changes

import cv2
model = YOLO("Services/intrusion_detection/resources/weights/yolov8n.engine", task="detect")  # load an official model
from sys import path,argv
from pathlib import Path
from threading import Thread
path.append(str(Path(__file__).resolve().parents[2]))
import json
service_name_dir = str(Path(__file__).resolve().parts[-2])


config_json_path = os.path.join('Services',service_name_dir,'config.json')
SERVICE_NAME = get_config_json(config_json_path)['service_name']

send_version_and_classes_details(config_json_path,DJANGO_API_ENDPOINT)#changes


import time
stime = time.time()

#changes
########################             common API Endpoints  ######################################
@app.route('/update_pipeline_service_data/',methods=['POST'])
def update_pipeline_service_data():
    global CAMERAS_DATA
    try:
        
        SLEEP_TIME = 3
        data = request.get_json()
        # print(data)

        # CAMERAS_DATA = {}
        if type(data['path']) == str:
            data['path'] = [data['path']]

        for i, path in enumerate(data['path']):
            path_components = path.strip('[]').split('][')
            idx = int(path_components[0])
            if idx not in CAMERAS_DATA:
                CAMERAS_DATA[idx] = {}
                cam_data = fetch_camera_data(idx,DJANGO_API_ENDPOINT)

                CAMERAS_DATA[idx]["tracker"] = Sort(max_age=1, min_hits=0, iou_threshold=0.1)
                CAMERAS_DATA[idx]["dict_add_id"] = {}
                CAMERAS_DATA[idx]["dict_add_id_known"] = {}
                CAMERAS_DATA[idx]["events_list"] = {}
                CAMERAS_DATA[idx]["filter"] = {}
                CAMERAS_DATA[idx]["skip_counter"] = 0
                CAMERAS_DATA[idx]["count_check_dict_in"] = {}#changes
                CAMERAS_DATA[idx]["active_status"] = True#changes
                
                if cam_data:
                    CAMERAS[idx] = cam_data
                
            current_level = CAMERAS_DATA[idx]
            if len(path_components) > 1:
                if 'relation_fields' not in current_level:
                    current_level['relation_fields'] = {}
                current_level = current_level['relation_fields']
                for component in path_components[1:]:
                    if component not in current_level:
                        current_level[component] = {}
                    current_level = current_level[component]
                try:
                    current_level[data['object_id']].update(data['changed_fields'])
                except:
                    current_level[data['object_id']] = data['changed_fields']
                    
                print(f"if block")
            else:
                print(f"else block")
                CAMERAS_DATA[idx].update(data['changed_fields'])

        
            if "extra_fields" not in CAMERAS_DATA[idx]:
                CAMERAS_DATA[idx]["extra_fields"] = {}

            if "service_activation_schedule" not in CAMERAS_DATA[idx]:
                CAMERAS_DATA[idx]["service_activation_schedule"] = []

        SLEEP_TIME = 0

        res = ''
        for key in CAMERAS_DATA:
            res += f'{CAMERAS_DATA[key]}<br><br>'

        return f"""
            {'*'*25 + '  Service_Name = ' + SERVICE_NAME + '  ' + '*'*25}<br>
            CAMERAS = {CAMERAS} <br>
            CAMERAS_DATA = {CAMERAS_DATA} <br>
            SLEEP_TIME = {SLEEP_TIME} <br>
        
        """

    except Exception as e:
        print(f"update_pipeline_service_data Exception occured due to {e}")
        SLEEP_TIME = 0


def get_all_service_cameras(DJANGO_API_ENDPOINT,SERVICE_NAME):
    try:
        while True:
            try:
                SLEEP_TIME = 5
                global CAMERAS_DATA,CAMERAS
                data=post(f'{DJANGO_API_ENDPOINT}/service_app/get_data/',json={"class_name":"CameraServiceConfig","service_name":SERVICE_NAME},headers=get_api_headers()).json()
                print(f"data = {data} and SERVICE_NAME = {SERVICE_NAME}")
                for i in data:
                    try:
                        CAMERAS_DATA[i['camera_id']]=i
                        CAMERAS_DATA[i['camera_id']]["tracker"] = Sort(max_age=1, min_hits=0, iou_threshold=0.1)
                        CAMERAS_DATA[i['camera_id']]["dict_add_id"] = {}
                        CAMERAS_DATA[i['camera_id']]["dict_add_id_known"] = {}
                        CAMERAS_DATA[i['camera_id']]["events_list"] = {}
                        CAMERAS_DATA[i['camera_id']]["filter"] = {}
                        CAMERAS_DATA[i['camera_id']]["skip_counter"] = 0
                        CAMERAS_DATA[i['camera_id']]["count_check_dict_in"] = {}#changes
                        CAMERAS_DATA[i['camera_id']]["active_status"] = True#changes

                        CAMERAS_DATA[i['camera_id']]["extra_fields"] = i['extra_fields']
                        
                        cam_data = fetch_camera_data(i['camera_id'],DJANGO_API_ENDPOINT)
                        print(f"cam_data = {cam_data}")
                        if cam_data:
                            CAMERAS[i['camera_id']] = cam_data
                    except:
                        pass

                print(CAMERAS_DATA)
                break
            
            except:
                pass

        
    except Exception as e:
        print(f"get_all_service_cameras error sue to {e}")
    SLEEP_TIME = 0

########################             /common API Endpoints  #####################################

#################################### FR Custom API's #############################################
@app.route("/multiple_angle_face_validation/", methods=["POST"])
def multiple_angle_face_validation():
    try:
        SLEEP_TIME = 5
        data = request.get_json()
        byte_data = b64decode(data["capturedFace"])
        face = cv2.imdecode(frombuffer(byte_data, uint8), cv2.IMREAD_COLOR)
        response, valid_face = face_angle(face, data["faceAngle"])
        # response , valid_face = 'accepted', True
        if valid_face:
            SLEEP_TIME = 0

            return Response(f"{response}", status=200, mimetype="application/json")
        else:
            SLEEP_TIME = 0
            return Response(f"{response}", status=400, mimetype="application/json")
    except Exception as e:
        SLEEP_TIME = 0
        # pipeline_error_logger.error(f"function_name : face_angle_verification , Error : {e}")
        return Response("Error occured at the multiple_angle_face_validation", status=400, mimetype="application/json")

#---------------------------------------------------------------------------
def create_image_encoding(img, encoding_version=1):
    input_shape_y, input_shape_x = 160, 160

    if encoding_version == 1:
        try:
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return face_recognition.face_encodings(img)[0]
        except:
            pass

    if encoding_version >= 2:
        try:
            boxes, _, _ = predictor.predict(img, candidate_size / 2, threshold1)
            x1, y1, x2, y2 = boxes[0]
            img = img[int(y1) : int(y2), int(x1) : int(x2)]
            img = cv2.resize(img, (input_shape_y, input_shape_x))

            img = np.expand_dims(img, axis=0)

            mean, std = img.mean(), img.std()
            img = (img - mean) / std

            # img_representation = optimized_model_reco(torch.from_numpy(img))[0][0, :]
            # img_representation = optimized_model_reco(img)[0][0, :]
            # img_representation = optimized_model_reco(img)[0][0]
            # img1_representation = optimized_model_reco.run(None, {'input_1': inp.astype(np.float32)})[0][0]
            img_representation = optimized_model_reco(img)
            # img_representation = img_representation
            print("img_representation",img_representation.shape)

            # img_representation = optimized_model_reco.run(None, {optimized_model_reco_name: img.astype(np.float32)})[0][0]
            # print(img_representation.tobytes())
            # print(f"{img_representation} and shape = {img_representation.dtype} type = {type(img_representation)}")
            # x = img_representation.numpy().tobytes()
            x = img_representation.tobytes()
            return x
        except Exception as e:
            print(f"Exception is {e}")
            return f"Exception is {e}"

#---------------------------------------------------------------------
@app.route("/create_face_embedding/", methods=["POST"])
def create_face_embedding():
    try:
        SLEEP_TIME = 5
        image_path = request.files["file"].read()
        # print(image_path)
        image = cv2.imdecode(frombuffer(image_path, uint8), cv2.IMREAD_COLOR)
        face_encode = str(create_image_encoding(image, encoding_version=2))
        # face_encode='functionality not implemented'
        return {"face_encode": face_encode}
    except:
        SLEEP_TIME = 0


#################################### /FR Custom API's ############################################
#changes

if __name__ == "__main__":
    try:
       
        #1. Initialzing the HTTP SERVER
        data = {'host':"0.0.0.0",'port':8080,'service_name':SERVICE_NAME}
        http_server = Thread(target=run_http_server,args=(app,data,))
        http_server.start()

        get_all_service_cameras(DJANGO_API_ENDPOINT,SERVICE_NAME)#changes

        get_and_update_all_service_timings(DJANGO_API_ENDPOINT,SERVICE_NAME)
        print(f"SERVICE_NAME = {SERVICE_NAME} CAMERAS_DATA = {CAMERAS_DATA}")

        # get_all_service_cameras()

        SERVICE_MANAGER_PRESENT = True
        #2. Service  Code Goes Here
        # frame_counter = 0
        global SERVICE_CONFIG_DATA
        SERVICE_CONFIG_DATA  = set_parameter_with_servce_name(DJANGO_API_ENDPOINT)#changes

        # SERVICE_CONFIG_DATA  = set_parameter_with_servce_name()
        for _ in range(20):
            model.predict(np.random.rand(1290,2000,3) ,classes=0,verbose=False, imgsz=[736,1280],conf = 0.7)[0].boxes.cpu().numpy().data
        print("----------------------warm up done ------------------------")

        while True:
            try:
                if time.time()-stime > 200:
                    SERVICE_MANAGER_PRESENT = check_heart_beat(DJANGO_API_ENDPOINT = DJANGO_API_ENDPOINT)#changes

                    # SERVICE_MANAGER_PRESENT = check_heart_beat()
                    stime = time.time()
                if SERVICE_MANAGER_PRESENT:
                

                    for camera_id in CAMERAS:
                        try:
                            if 'active_status' in CAMERAS_DATA[camera_id] and not CAMERAS_DATA[camera_id]['active_status']:
                                continue

                            if CAMERAS_DATA[camera_id]["skip_counter"] % int(SERVICE_CONFIG_DATA["every_n_frame"]) == 0:
                                ret,frame = CAMERAS[camera_id]['cap'].read()
                                # print(SERVICE_CONFIG_DATA["every_n_frame"],"###############################")


                                h_org, w_org, _ = frame.shape
                                ui_corrd_width = 1090#CAMERAS_DATA[camera_id]["extra_fields"]['image_width']
                                ui_corrd_height = 452#CAMERAS_DATA[camera_id]["extra_fields"]['image_height']
                                scaley = h_org / ui_corrd_height
                                scalex = w_org / ui_corrd_width
                                
                                if "confidence_score" in CAMERAS_DATA[camera_id]["extra_fields"]:
                                    conf = CAMERAS_DATA[camera_id]["extra_fields"]["confidence_score"]
                                else:
                                    conf= 0.7

                                emty_numpy_array = model.predict(frame,classes=0,verbose=False, imgsz=[736,1280],conf = conf)[0].boxes.cpu().numpy().data

                                track_bbs_ids = CAMERAS_DATA[camera_id]["tracker"].update(emty_numpy_array)
                                (frame,give_event,bbox_output,image_selection_bbox) = counting_in_out_person_test(
                                    track_bbs_ids,
                                    frame,
                                    CAMERAS_DATA[camera_id]["extra_fields"],
                                    scalex,
                                    scaley,
                                    CAMERAS_DATA[camera_id]["count_check_dict_in"]
                                )



                                if len(CAMERAS_DATA[camera_id]["count_check_dict_in"]) > 150:
                                    del CAMERAS_DATA[camera_id]["count_check_dict_in"][list(CAMERAS_DATA[camera_id]["count_check_dict_in"].keys())[0]]
                                if give_event:
                                    
                                    service_id = CAMERAS_DATA[camera_id]['service_id']
                                    user_id = CAMERAS_DATA[camera_id]['user_id']
                                    camera_name = CAMERAS[camera_id]['camera_name']
                                    service_name = CAMERAS_DATA[camera_id]['service_name']
                                    alert = Alert(f"Intrusion detected",camera_id = camera_id,service_id = service_id,user_id = user_id,camera_name=camera_name,service_name=service_name,ALERT_SEND_URL = ALERT_SEND_URL)#changes
                                    
                                    # alert = Alert(f"Intrusion detected",camera_id = camera_id,service_id = service_id,user_id = user_id,camera_name=camera_name,service_name=service_name)
                                    # bbox_output
                                    res_bbox = {'bbox':bbox_output,'image_size':[h_org, w_org],'image_selection_bbox':image_selection_bbox}
                                    alert.bbox = json.dumps([res_bbox])

                                    class_data = []
                                    class_data.append({
                                        "class_name":"Intrusion","value":True,"value_type":"bool"
                                    })
                                    alert.classes = json.dumps(class_data)
                                    alert.message_details = json.dumps(["Intrusion detected"])

                                    alert.send(frame)
                                    # print(f"Alert Send {alert.__dict__}")
                                    

                                if CAMERAS[camera_id]['is_watch']:
                                    cv2.imshow(f"{camera_id}", frame) 

                                    cv2.waitKey(1)
                            else:
                                ret = CAMERAS[camera_id]["cap"].grab()
                            CAMERAS_DATA[camera_id]["skip_counter"] += 1

                            
                            if CAMERAS_DATA[camera_id]["skip_counter"] >= 100000:
                                # SERVICE_MANAGER_PRESENT = check_heart_beat()
                                CAMERAS_DATA[camera_id]["skip_counter"] = 0

                        except Exception as e:
                            if not ret:
                                CAMERAS[camera_id]['cap'] =  cv2.VideoCapture(CAMERAS[camera_id]['rtsp_url']) 
                            print(traceback.format_exc())
                            print(f"unable to process camera of camera_id = {camera_id} due to {e}")
                
                # CAMERAS_DATA[camera_id]["skip_counter"] += 1

                if len(CAMERAS) == 0 or not SERVICE_MANAGER_PRESENT:
                    sleep(3)

                # if CAMERAS_DATA[camera_id]["skip_counter"] >= 100000:
                #     # SERVICE_MANAGER_PRESENT = check_heart_beat()
                #     CAMERAS_DATA[camera_id]["skip_counter"] = 0

                sleep(SLEEP_TIME)


            except :
                print(traceback.format_exc())
                print(f"Program Exited...")

                

    except Exception as e:
        print(traceback.format_exc())

        print(f"unable to run the {SERVICE_NAME} program due to {e}")
