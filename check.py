# from google import genai
# from PIL import Image

# #Initialize 
# client = genai.Client(api_key="YOUR KEY")

# INPUT_IMAGE_PATH = "cropimg.png"
# OUTPUT_IMAGE_PATH = "annotated_path.png"

# def annotate_floor_plan(origin, destination):
#     # Load the floor plan
#     floor_plan = Image.open(INPUT_IMAGE_PATH)

#     # Create prompt
#     prompt = (
#         f"You are an indoor navigation expert. Given this floor plan, "
#         f"draw a clear, bold red path from room {origin} to room {destination} and return the image. "
#     )

#     # Call the Gemini model
#     response = client.models.generate_content(
#         model="gemini-2.5-flash-image",
#         contents=[prompt, floor_plan],
#     )

#     # Extract image from response
#     for part in response.parts:
#         if part.inline_data is not None:
#             annotated_image = part.as_image()
#             annotated_image.save(OUTPUT_IMAGE_PATH)
#             print(f"Annotated floor plan saved to {OUTPUT_IMAGE_PATH}")
#             return

#     print(" No image returned from Gemini.")

# def main():
#     origin = input("Enter origin room: ").strip()
#     destination = input("Enter destination room: ").strip()
#     if not origin or not destination:
#         print("Both origin and destination are required.")
#         return

#     annotate_floor_plan(origin, destination)

# if __name__ == "__main__":
#     main()


from google import genai
from google.genai import types
from PIL import Image

API_KEY = "AIzaSyB4akC2LeTmRn7551CoVEQNXaegcKx9ypw"

def test_gemini_image():
    try:
        # Initialize client
        client = genai.Client(api_key=API_KEY)

        prompt = "Draw a simple red circle on a white background."

        print("Sending test request to Gemini...")

        # Request image generation
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt],
        )

        # Process result
        image_found = False
        for part in response.parts:
            if part.inline_data is not None:
                img = part.as_image()
                img.save("gemini_test_output.png")
                print("Image saved as gemini_test_output.png")
                image_found = True

        if not image_found:
            print("No image data returned. Check if your key supports image models.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gemini_image()
