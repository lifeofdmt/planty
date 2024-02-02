from utility import program_parser, load_cat, configure_device, imshow, process_image
from helper import load_checkpoint, prediction

parser = program_parser(train=False)
args = parser.parse_args()

# Configure training device
device = configure_device(args.gpu)

# Store category to name mapping
if args.category_names != None:
    args.category_names = load_cat(args.category_names)
    
# Make prediction
top_ps, top_classes = prediction(image_path=args.input, model_checkpoint=args.checkpoint,
                                 k=args.top_k, device=device, cat_to_name=args.category_names)
print()
print("Class Probabilities")
print(top_ps)
print()
print("Most Probable Classes")
print(top_classes)
print()

