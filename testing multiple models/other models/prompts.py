"""
50 Creative prompts for benchmark testing
"""

PROMPTS = [
    "An elegant owl librarian wearing reading glasses.",
    "A red fox sitting at a pottery wheel, shaping a clay vase with its paws.",
    "A majestic eagle wearing a pilot's headset, flying a vintage biplane through the clouds.",
    "A squirrel wearing a business suit, aggressively speaking into a tiny gold smartphone.",
    "An astronaut riding a bicycle on a ring of Saturn, delivering a cardboard pizza box.",
    "A fluffy kitten wearing heavy metal spiked armor.",
    "A cat wearing headphones DJing at a packed nightclub.",
    "A golden retriever in a tuxedo swimming.",
    "A hamster lifting a tiny barbell in a gym.",
    "A squirrel wearing a cowboy hat riding a skateboard.",
    "A penguin wearing a grass skirt doing the hula dance.",
    "A pug wearing sunglasses sunbathing on a pool float.",
    "A pigeon eating a tiny slice of pizza with a fork.",
    "A frog playing a miniature electric guitar on a lilypad.",
    "A fancy owl drinking a cup of hot coffee.",
    "A baby monkey wearing a diaper and riding a tricycle.",
    "Spider-Man sitting on a park bench eating a banana.",
    "Darth Vader aggressively vacuuming a living room rug.",
    "Batman crying while chopping a massive onion.",
    "Shrek doing a delicate ballet dance on a stage.",
    "An astronaut playing the bagpipes on the moon.",
    "A medieval knight trying to use a modern smartphone.",
    "A scary T-Rex wearing a pink frilly apron baking cookies.",
    "Superman getting a painful leg wax at a salon.",
    "Iron Man attempting to knit a colorful winter scarf.",
    "A mummy covered in bandages getting stuck in a revolving door.",
    "A slice of pepperoni pizza skateboarding down a handrail.",
    "A giant angry taco lifting weights in a gym.",
    "An avocado playing a tiny set of drums.",
    "A banana slipping and falling on a human banana peel.",
    "A marshmallow lifting weights over a campfire.",
    "A sad hotdog sitting alone at a fancy restaurant table.",
    "A box of french fries taking a selfie together.",
    "A sentient alarm clock screaming and running away.",
    "A walking toaster throwing slices of burnt toast like boomerangs.",
    "A group of broccoli pieces having a loud rock concert.",
    "A business suit made entirely out of bubble wrap popping.",
    "A toilet paper roll escaping down a busy city street.",
    "A giant rubber ducky driving a red convertible car.",
    "A cloud in the sky blowing bubble gum.",
    "A couch sitting on a human living room floor.",
    "A professional office meeting run entirely by plush teddy bears.",
    "A lawnmower cutting a lawn made entirely of green spaghetti.",
    "A grandfather clock walking on two robotic legs.",
    "A computer monitor wearing reading glasses and looking confused.",
    "A baby rattle shaking itself to heavy metal music.",
    "Cat riding a vacuum cleaner.",
    "Banana playing the saxophone.",
    "Dinosaur on a pogo stick.",
    "Robot eating a bowl of cereal.",
    "Pig flying a helicopter.",
    "Shark wearing a party hat.",
    "Chicken lifting heavy weights.",
    "Potato singing opera on stage.",
    "Alien doing the chicken dance.",
    "Ninja turtle eating ice cream.",
]

def get_prompts():
    """Return all 50 prompts"""
    return PROMPTS

def get_prompt_by_id(prompt_id):
    """Get a specific prompt by ID (1-50)"""
    if 1 <= prompt_id <= len(PROMPTS):
        return PROMPTS[prompt_id - 1]
    else:
        raise ValueError(f"Prompt ID must be between 1 and {len(PROMPTS)}")

def get_total_prompts():
    """Return the total number of prompts"""
    return len(PROMPTS)
