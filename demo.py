from argparse import ArgumentParser
from demo_api import image, video, live
from model.C_CDN import C_CDN, DC_CDN
from model.CDCN import CDCN



map_input_to_model = {'C_CDN': C_CDN, 'DC_CDN': DC_CDN, 'CDCN': CDCN}
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i','--image', required= False, help= 'Detect liveness score from a single image')
    parser.add_argument('-v', '--video', required= False, help= 'Detect liveness score in a video')
    parser.add_argument('--model', default= DC_CDN, help= 'Decide what model to use')
    parser.add_argument('--model_weights', default= 'checkpoints/checkpoint_dc_cdn_mix.pth', help= 'Use checkpoint that compatible with the model')
    parser.add_argument('--fps', default=15, help= 'Decide maximum FPS limit for video and live cam function.')
    parser.add_argument('--check_rate', default= 1., help= 'Deicde how many second(s) will be between each check turn.')
    parser.add_argument('-m', '--max_faces',default= -1, help= 'Decide the maximum of face will be detect for each image/frame.')
    parser.add_argument('-w', '--max_workers',default= 2, help= 'Decide the maximum number of parallel process(es) will be used. ')
    parser.add_argument('--theta', default= .7, type= float, help= 'Decide the trade-off between vanilla convolution and central difference convolution.')
    # parser.add_argument()
    args = parser.parse_args()
    # model = map_input_to_model[args.model]
    try: 
        model = args.model
        model()
    except:
        model = map_input_to_model[args.model]
    if args.image is not None:
        image(args.image, maxFaceDetected= args.max_faces, maxWorkers= args.max_workers, model = model(theta= args.theta),modelDepthWeight= args.model_weights)
        
    elif args.video is not None:
        try:
            video(args.video,fpsLimit= args.fps,faceCheckRate= args.check_rate, maxFaceDetected= args.max_faces, maxWorkers= args.max_workers, model = model(theta= args.theta),modelDepthWeight= args.model_weights)
        except Exception as e:
            print(e)
            exit()
    else:
        live(fpsLimit= args.fps,faceCheckRate= args.check_rate,maxFaceDetected= args.max_faces, maxWorkers= args.max_workers, model = model(theta= args.theta),modelDepthWeight= args.model_weights)