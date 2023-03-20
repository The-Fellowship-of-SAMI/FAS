import torch
import cv2
from model.Finetune import Finetune_model
from model.C_CDN import C_CDN, DC_CDN
from model.CDCN import CDCN, ATT_CDCN
import dlib
import time
import threading
import os
from model import pl_model

def validate(x,i):
    # global score
    
    live_score = val_model(x.cuda()).cpu().detach().float().numpy()[0]
    # plt.imshow(depth_map, cmap = 'gray')
    # plt.show()
    score[i] = round(live_score[0][0],3)
    # print(score[i])

def validate_new(x,i):
    # global score
    
    live_score = val_model(x.cuda())[1].cpu().detach().float().numpy()
    # plt.imshow(depth_map, cmap = 'gray')
    # plt.show()
    score[i] = round(live_score[0][1],3)
    # print(score[i])



def image(image, maxFaceDetected = -1, maxWorkers = 2, model = C_CDN(), modelDepthWeight = 'checkpoints/checkpoint_c_cdn_mix.pth'):
    assert maxFaceDetected == -1 or maxFaceDetected >0 , "Faces detected should be higher than zero."
    assert maxWorkers == -1 or maxWorkers > 1 , "Number of workers should be higher than zero."
    
    global val_model
    global score

    score = {}
    val_model = Finetune_model(depth_model=model, depth_weights= modelDepthWeight).cuda()
    val_model.cls.load_state_dict(torch.load('checkpoints/checkpoint_cls_sm.pth'))
    val_model.eval()
    
    frame = cv2.imread(image)
    h, w, _ = frame.shape
    scale = min(min(1920/w, 1080/h),1)
    detector = dlib.cnn_face_detection_model_v1('./depthgen/Data/net-data/mmod_human_face_detector.dat')
    i=0

    if scale != 1:
                frame = cv2.resize(frame,(int(w*scale), int(h*scale)))
        
    try:
            # global score
        detected_face = detector(frame,1)
        # print(len(detected_face))
        num_face_detect = min(len(detected_face), maxFaceDetected) if maxFaceDetected != -1 else len(detected_face)
        num_workers = num_face_detect if maxWorkers == -1 else maxWorkers
        # score = list(np.zeros(num_face_detect))

        l = list(range(num_face_detect))
        r = list(range(num_face_detect))
        t = list(range(num_face_detect))
        b = list(range(num_face_detect))
        
        w_b = list(range(num_face_detect))
        h_b = list(range(num_face_detect))
        for i in range(num_face_detect):
            bbox = detected_face[i].rect
            l[i],r[i],t[i],b[i] = bbox.left(), bbox.right(), bbox.top(), bbox.bottom()
            w_b[i] = r[i] - l[i]
            h_b[i] = b[i] - t[i]
            
            # print(f'{l[i]} {r[i]} {t[i]} {b[i]}')
             
        if num_workers == num_face_detect:
                for i in range(num_face_detect):
                    
                    cropped_face = frame[max(t[i]- int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                    # print(r[i]-l[i], " ", b[i]- t[i])
                    # print(cropped_face.shape)
                    cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256), interpolation= cv2.INTER_CUBIC) ).permute(2,0,1).unsqueeze(0)/255
                    ### add face anti-spoofing code here  ###
                    # print(np.shape(cropped_face))
                    
                    t1 = threading.Thread(target=validate, args=(cropped_face,i,))
                    t1.start()
                
                t1.join()
        else:
                num_iter = num_face_detect//num_workers
                res_iter = num_face_detect%num_workers

                for cur_iter in range(num_iter):
                    for i in range(num_workers):
                        cropped_face = frame[max(t[i]-int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                        # print(r[i]-l[i], " ", b[i]- t[i])
                        cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255   
                        t1 = threading.Thread(target=validate, args=(cropped_face,cur_iter*num_workers +i,))
                        t1.start()
                    t1.join()

                for i in range(res_iter):
                    cropped_face = frame[max(t[i]-int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                    # print(r[i]-l[i], " ", b[i]- t[i])
                    cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255
                    t1 = threading.Thread(target=validate, args=(cropped_face,num_iter*num_workers +i,))
                    t1.start()
                
                t1.join()

        for i in range(num_face_detect):
            # print(ct)
            frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.2*h_b[i]),0)),(min(r[i]+int(.1*w_b[i]),w),min(b[i]+int(.2*h_b[i]),h)), color = (255, 0, 0), thickness= 2)
            frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.35*h_b[i]),0)),(max(r[i]-int(.7*w_b[i]),0),max(t[i]-int(.2*h_b[i]),int(.1*h_b[i]))), color = (255, 0, 0), thickness= -1)

            frame = cv2.putText(frame,str(score[i]),org = (max(l[i]-int(.07*w_b[i]),0),max(t[i]-int(.25*h_b[i]),int(.1*h_b[i]))),\
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = (b[i]-t[i])/250, color = (255, 255, 255), thickness = int(2*(b[i]-t[i])/250),lineType= cv2.LINE_AA)
        
    except Exception as e:
        print(e)
    finally:
        cv2.imshow('frame', frame)


    if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()


def live( fpsLimit = 15, faceCheckRate = 1., maxFaceDetected = -1, maxWorkers = 2, model = C_CDN(), modelDepthWeight = 'checkpoints/checkpoint_c_cdn_mix.pth'): 
    
    assert maxFaceDetected == -1 or maxFaceDetected >0 , "Faces detected should be higher than zero."
    assert maxWorkers == -1 or maxWorkers > 1 , "Number of workers should be higher than zero."
    
    global val_model
    global score 

    score = {}
    val_model = Finetune_model(depth_model=model, depth_weights= modelDepthWeight).cuda()
    val_model.cls.load_state_dict(torch.load('checkpoints/checkpoint_cls_sm.pth'))
    val_model.eval()
    
    vid = cv2.VideoCapture(0)
    w, h = vid.get(3), vid.get(4)
    detector = dlib.cnn_face_detection_model_v1('./depthgen/Data/net-data/mmod_human_face_detector.dat')
    i=0
    ct = time.time()
    fct = time.time()
    while(True):
        
        # Capture the video frame by frame
        ret, frame = vid.read()
        # Display the resulting 
        if ret:
            if time.time()- ct > 1./fpsLimit:
                
                try:
                    # global score
                    detected_face = detector(frame,1)
                    
                    _ = detected_face[0] # call this dummny line for exception when there is no face
                    num_face_detect = min(len(detected_face), maxFaceDetected) if maxFaceDetected != -1 else len(detected_face)
                    num_workers = num_face_detect if maxWorkers == -1 else maxWorkers
                    # score = list(np.zeros(num_face_detect))

                    l = list(range(num_face_detect))
                    r = list(range(num_face_detect))
                    t = list(range(num_face_detect))
                    b = list(range(num_face_detect))
                    
                    w_b = list(range(num_face_detect))
                    h_b = list(range(num_face_detect))
                    for i in range(num_face_detect):
                        bbox = detected_face[i].rect
                        l[i],r[i],t[i],b[i] = bbox.left(), bbox.right(), bbox.top(), bbox.bottom()
                        w_b[i] = r[i] - l[i]
                        h_b[i] = b[i] - t[i]
                        
                        # print(f'{l} {r} {t} {b}')
                        
                    if time.time() - fct >faceCheckRate:
                        fct = time.time()
                        
                        if num_workers == num_face_detect:
                            for i in range(num_face_detect):
                               
                                cropped_face = frame[max(t[i]- int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                                # print(r[i]-l[i], " ", b[i]- t[i])
                                cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256), interpolation= cv2.INTER_CUBIC) ).permute(2,0,1).unsqueeze(0)/255
                                ### add face anti-spoofing code here  ###
                                # print(np.shape(cropped_face))
                                
                                t1 = threading.Thread(target=validate, args=(cropped_face,i,))
                                t1.start()
                            
                            t1.join()
                        else:
                            num_iter = num_face_detect//num_workers
                            res_iter = num_face_detect%num_workers

                            for cur_iter in range(num_iter):
                                for i in range(num_workers):
                                    cropped_face = frame[max(t[i]-int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                                    # print(r[i]-l[i], " ", b[i]- t[i])
                                    cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255   
                                    t1 = threading.Thread(target=validate, args=(cropped_face,cur_iter*num_workers +i,))
                                    t1.start()
                                t1.join()

                            for i in range(res_iter):
                                cropped_face = frame[max(t[i]-int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                                # print(r[i]-l[i], " ", b[i]- t[i])
                                cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255
                                t1 = threading.Thread(target=validate, args=(cropped_face,num_iter*num_workers +i,))
                                t1.start()
                            
                            t1.join()

                    for i in range(num_face_detect):
                        # print(ct)
                        frame = cv2.rectangle(frame, pt1= (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.2*h_b[i]),0)),pt2= (min(r[i]+int(.1*w_b[i]),w),min(b[i]+int(.2*h_b[i]),int(h))), color = (255, 0, 0), thickness= 2)
                        frame = cv2.rectangle(frame, pt1= (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.35*h_b[i]),0)),pt2= (max(r[i]-int(.7*w_b[i]),0),max(t[i]-int(.2*h_b[i]),int(.1*h_b[i]))), color = (255, 0, 0), thickness= -1)

                        frame = cv2.putText(frame,str(score[i]),org = (max(l[i]-int(.07*w_b[i]),0),max(t[i]-int(.25*h_b[i]),int(.1*h_b[i]))),\
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = (b[i]-t[i])/250, color = (255, 255, 255), thickness = int(2*(b[i]-t[i])/250),lineType= cv2.LINE_AA)
                    
                except Exception as e:
                    # print("Exception occured:",e)
                    if time.time() - fct >faceCheckRate:
                        fct = time.time()
                        cropped_face = frame[:,:,:]
                        cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255
                        ### add face anti-spoofing code here  ###
                        # print(cropped_face.shape)
                        
                        t2 = threading.Thread(target=validate, args=(cropped_face,-1,))
                        t2.start()
                        
                        t2.join()
                        

                        # frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.2*h_b[i]),0)),(min(r[i]+int(.1*w_b[i]),w),min(b[i]+int(.2*h_b[i]),h)), color = (255, 0, 0), thickness= 2)
                        # frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.35*h_b[i]),0)),(max(r[i]-int(.7*w_b[i]),0),max(t[i]-int(.2*h_b[i]),int(.1*h_b[i]))), color = (255, 0, 0), thickness= -1)
                        # print(score[-1])
                    try:
                        frame = cv2.putText(frame,str(score[-1]),org = (0,20),\
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color = (255, 255, 255), thickness = 2,lineType= cv2.LINE_AA)
                    except:
                        pass
                    
                finally:
                    cv2.imshow('frame', frame)
                    ct = time.time()
            # plt.imshow(frame[::-1])
            # plt.show()
        if cv2.waitKey(1) == ord('q'):
            break
        # the 'q' button is set as the quitting button
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def video(video , fpsLimit = 15, faceCheckRate = 1., maxFaceDetected = -1, maxWorkers = 2, model = C_CDN(), modelDepthWeight = 'checkpoints/checkpoint_c_cdn_mix.pth'): 
    
    assert maxFaceDetected == -1 or maxFaceDetected >0 , "Faces detected should be higher than zero."
    assert maxWorkers == -1 or maxWorkers > 1 , "Number of workers should be higher than zero."
    
    global val_model
    global score 

    score = {}
    val_model = Finetune_model(depth_model=model, depth_weights= modelDepthWeight).cuda()
    val_model.cls.load_state_dict(torch.load('checkpoints/checkpoint_cls_sm.pth'))
    val_model.eval()
    
    if not os.path.isfile(video):
        raise Exception("Enter valid file or directory name.")
    vid = cv2.VideoCapture(video)
    w, h = vid.get(3), vid.get(4)
    scale = min(min(1920/w, 1080/h),1)
    
    detector = dlib.cnn_face_detection_model_v1('./depthgen/Data/net-data/mmod_human_face_detector.dat')
    i=0
    ct = time.time()
    fct = time.time()
    while(True):
        
        # Capture the video frame by frame
        
        ret, frame = vid.read()
        
        # Display the resulting 
        if ret:
            if scale != 1:
                frame = cv2.resize(frame,(int(w*scale), int(h*scale)))
            if time.time()- ct > 1./fpsLimit:
                
                try:
                    # global score
                    detected_face = detector(frame,1)
                    _ = detected_face[0] # call this dummny line for exception when there is no face

                    num_face_detect = min(len(detected_face), maxFaceDetected) if maxFaceDetected != -1 else len(detected_face)
                    num_workers = num_face_detect if maxWorkers == -1 else maxWorkers
                    # score = list(np.zeros(num_face_detect))

                    l = list(range(num_face_detect))
                    r = list(range(num_face_detect))
                    t = list(range(num_face_detect))
                    b = list(range(num_face_detect))
                    
                    w_b = list(range(num_face_detect))
                    h_b = list(range(num_face_detect))
                    for i in range(num_face_detect):
                        bbox = detected_face[i].rect
                        l[i],r[i],t[i],b[i] = bbox.left(), bbox.right(), bbox.top(), bbox.bottom()
                        w_b[i] = r[i] - l[i]
                        h_b[i] = b[i] - t[i]
                        
                        # print(f'{l} {r} {t} {b}')
                        
                    if time.time() - fct >faceCheckRate:
                        fct = time.time()
                        
                        if num_workers == num_face_detect:
                            for i in range(num_face_detect):
                               
                                cropped_face = frame[max(t[i]- int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                                # print(r[i]-l[i], " ", b[i]- t[i])
                                cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256), interpolation= cv2.INTER_CUBIC) ).permute(2,0,1).unsqueeze(0)/255
                                ### add face anti-spoofing code here  ###
                                # print(np.shape(cropped_face))
                                
                                t1 = threading.Thread(target=validate, args=(cropped_face,i,))
                                t1.start()
                            
                            t1.join()
                        else:
                            num_iter = num_face_detect//num_workers
                            res_iter = num_face_detect%num_workers

                            for cur_iter in range(num_iter):
                                for i in range(num_workers):
                                    cropped_face = frame[max(t[i]-int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                                    # print(r[i]-l[i], " ", b[i]- t[i])
                                    cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255   
                                    t1 = threading.Thread(target=validate, args=(cropped_face,cur_iter*num_workers +i,))
                                    t1.start()
                                t1.join()

                            for i in range(res_iter):
                                cropped_face = frame[max(t[i]-int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                                # print(r[i]-l[i], " ", b[i]- t[i])
                                cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255
                                t1 = threading.Thread(target=validate, args=(cropped_face,num_iter*num_workers +i,))
                                t1.start()
                            
                            t1.join()

                    for i in range(num_face_detect):
                        # print(ct)
                        frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.2*h_b[i]),0)),(min(r[i]+int(.1*w_b[i]),w),min(b[i]+int(.2*h_b[i]),h)), color = (255, 0, 0), thickness= 2)
                        frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.35*h_b[i]),0)),(max(r[i]-int(.7*w_b[i]),0),max(t[i]-int(.2*h_b[i]),int(.1*h_b[i]))), color = (255, 0, 0), thickness= -1)

                        frame = cv2.putText(frame,str(score[i]),org = (max(l[i]-int(.07*w_b[i]),0),max(t[i]-int(.25*h_b[i]),int(.1*h_b[i]))),\
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = (b[i]-t[i])/250, color = (255, 255, 255), thickness = int(2*(b[i]-t[i])/250),lineType= cv2.LINE_AA)
                    
                except Exception as e:
                    # pass               
                    if time.time() - fct >faceCheckRate:
                        fct = time.time()
                        cropped_face = frame[:,:]
                        # print(r[i]-l[i], " ", b[i]- t[i])
                        cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255
                        ### add face anti-spoofing code here  ###
                        # print(cropped_face.shape)
                        
                        t2 = threading.Thread(target=validate, args=(cropped_face,-1,))
                        t2.start()
                        
                        t2.join()
                        

                        # frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.2*h_b[i]),0)),(min(r[i]+int(.1*w_b[i]),w),min(b[i]+int(.2*h_b[i]),h)), color = (255, 0, 0), thickness= 2)
                        # frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.35*h_b[i]),0)),(max(r[i]-int(.7*w_b[i]),0),max(t[i]-int(.2*h_b[i]),int(.1*h_b[i]))), color = (255, 0, 0), thickness= -1)
                        # print(score[0])
                    try:
                        frame = cv2.rectangle(frame, (0,0),(75,30), color = (255, 0, 0), thickness= -1)
                        # print(score[0])
                        frame = cv2.putText(frame,str(score[-1]),org = (0,20),\
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color = (255, 255, 255), thickness = 2,lineType= cv2.LINE_AA)
                    except:
                        pass
                        
                finally:
                    cv2.imshow('frame', frame)
                    ct = time.time()
            # plt.imshow(frame[::-1])
            # plt.show()
        if cv2.waitKey(1) == ord('q'):
            break
        # the 'q' button is set as the quitting button
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def live_new( fpsLimit = 15, faceCheckRate = 1., maxFaceDetected = -1, maxWorkers = 2):
    assert maxFaceDetected == -1 or maxFaceDetected >0 , "Faces detected should be higher than zero."
    assert maxWorkers == -1 or maxWorkers > 1 , "Number of workers should be higher than zero."
    
    global val_model
    global score 

    score = {}
    # val_model = Finetune_model(depth_model=model, depth_weights= modelDepthWeight).cuda()
    # val_model.cls.load_state_dict(torch.load('checkpoints/checkpoint_cls_sm.pth'))
    val_model = pl_model.load_from_checkpoint('checkpoints/checkpoint_att_cdcn_best.ckpt',model = ATT_CDCN(), train = 'all').cuda()

    val_model.eval()
    
    vid = cv2.VideoCapture(0)
    w, h = vid.get(3), vid.get(4)
    detector = dlib.cnn_face_detection_model_v1('./depthgen/Data/net-data/mmod_human_face_detector.dat')
    i=0
    ct = time.time()
    fct = time.time()
    while(True):
        
        # Capture the video frame by frame
        ret, frame = vid.read()
        # Display the resulting 
        if ret:
            if time.time()- ct > 1./fpsLimit:
                
                try:
                    # global score
                    detected_face = detector(frame,1)
                    
                    _ = detected_face[0] # call this dummny line for exception when there is no face
                    num_face_detect = min(len(detected_face), maxFaceDetected) if maxFaceDetected != -1 else len(detected_face)
                    num_workers = num_face_detect if maxWorkers == -1 else maxWorkers
                    # score = list(np.zeros(num_face_detect))

                    l = list(range(num_face_detect))
                    r = list(range(num_face_detect))
                    t = list(range(num_face_detect))
                    b = list(range(num_face_detect))
                    
                    w_b = list(range(num_face_detect))
                    h_b = list(range(num_face_detect))
                    for i in range(num_face_detect):
                        bbox = detected_face[i].rect
                        l[i],r[i],t[i],b[i] = bbox.left(), bbox.right(), bbox.top(), bbox.bottom()
                        w_b[i] = r[i] - l[i]
                        h_b[i] = b[i] - t[i]
                        
                        # print(f'{l} {r} {t} {b}')
                        
                    if time.time() - fct >faceCheckRate:
                        fct = time.time()
                        
                        if num_workers == num_face_detect:
                            for i in range(num_face_detect):
                               
                                cropped_face = frame[max(t[i]- int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                                # print(r[i]-l[i], " ", b[i]- t[i])
                                cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256), interpolation= cv2.INTER_CUBIC) ).permute(2,0,1).unsqueeze(0)/255
                                ### add face anti-spoofing code here  ###
                                # print(np.shape(cropped_face))
                                
                                t1 = threading.Thread(target=validate_new, args=(cropped_face,i,))
                                t1.start()
                            
                            t1.join()
                        else:
                            num_iter = num_face_detect//num_workers
                            res_iter = num_face_detect%num_workers

                            for cur_iter in range(num_iter):
                                for i in range(num_workers):
                                    cropped_face = frame[max(t[i]-int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                                    # print(r[i]-l[i], " ", b[i]- t[i])
                                    cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255   
                                    t1 = threading.Thread(target=validate_new, args=(cropped_face,cur_iter*num_workers +i,))
                                    t1.start()
                                t1.join()

                            for i in range(res_iter):
                                cropped_face = frame[max(t[i]-int(.2*h_b[i]),0):min(b[i]+int(.2*h_b[i]),h), max(l[i]-int(.1*w_b[i]),0): min(r[i]+int(.1*w_b[i]),w)]
                                # print(r[i]-l[i], " ", b[i]- t[i])
                                cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255
                                t1 = threading.Thread(target=validate_new, args=(cropped_face,num_iter*num_workers +i,))
                                t1.start()
                            
                            t1.join()

                    for i in range(num_face_detect):
                        # print(ct)
                        frame = cv2.rectangle(frame, pt1= (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.2*h_b[i]),0)),pt2= (min(r[i]+int(.1*w_b[i]),w),min(b[i]+int(.2*h_b[i]),int(h))), color = (255, 0, 0), thickness= 2)
                        frame = cv2.rectangle(frame, pt1= (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.35*h_b[i]),0)),pt2= (max(r[i]-int(.7*w_b[i]),0),max(t[i]-int(.2*h_b[i]),int(.1*h_b[i]))), color = (255, 0, 0), thickness= -1)

                        frame = cv2.putText(frame,str(score[i]),org = (max(l[i]-int(.07*w_b[i]),0),max(t[i]-int(.25*h_b[i]),int(.1*h_b[i]))),\
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = (b[i]-t[i])/250, color = (255, 255, 255), thickness = int(2*(b[i]-t[i])/250),lineType= cv2.LINE_AA)
                    
                except Exception as e:
                    # print("Exception occured:",e)
                    if time.time() - fct >faceCheckRate:
                        fct = time.time()
                        cropped_face = frame[:,:,:]
                        cropped_face = torch.Tensor(cv2.resize(cropped_face[:,:,::-1], (256,256)) ).permute(2,0,1).unsqueeze(0)/255
                        ### add face anti-spoofing code here  ###
                        # print(cropped_face.shape)
                        
                        t2 = threading.Thread(target=validate_new, args=(cropped_face,-1,))
                        t2.start()
                        
                        t2.join()
                        

                        # frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.2*h_b[i]),0)),(min(r[i]+int(.1*w_b[i]),w),min(b[i]+int(.2*h_b[i]),h)), color = (255, 0, 0), thickness= 2)
                        # frame = cv2.rectangle(frame, (max(l[i]-int(.1*w_b[i]),0),max(t[i]-int(.35*h_b[i]),0)),(max(r[i]-int(.7*w_b[i]),0),max(t[i]-int(.2*h_b[i]),int(.1*h_b[i]))), color = (255, 0, 0), thickness= -1)
                        # print(score[-1])
                    try:
                        frame = cv2.putText(frame,str(score[-1]),org = (0,20),\
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = .75, color = (255, 255, 255), thickness = 2,lineType= cv2.LINE_AA)
                    except:
                        pass
                    
                finally:
                    cv2.imshow('frame', frame)
                    ct = time.time()
            # plt.imshow(frame[::-1])
            # plt.show()
        if cv2.waitKey(1) == ord('q'):
            break
        # the 'q' button is set as the quitting button
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_new()