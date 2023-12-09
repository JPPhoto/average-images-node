# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)

import numpy as np
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from PIL import Image


@invocation("average_images", title="Average Images", tags=["image"], version="1.0.0")
class AverageImagesInvocation(BaseInvocation, WithMetadata):
    """Average images"""

    images: list[ImageField] = InputField(description="The collection of images to average")
    gamma: float = InputField(default=2.2, gt=0.0, description="Gamma for color correcting before/after blending")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        if len(self.images) == 0:
            raise ValueError("No input images specified")

        arr = None

        for this_image in self.images:
            image = context.services.images.get_pil_image(this_image.image_name)
            image = image.convert("RGB")
            if arr is None:
                w, h = image.size
                arr = np.zeros((h, w, 3), np.float32)

            imarr = np.array(image)
            imarr.astype(np.float32)
            imarr = (imarr / 255.0) ** (self.gamma)
            arr += imarr

        arr = arr / len(self.images)
        arr = np.clip(arr ** (1.0 / self.gamma), 0, 1) * 255.0
        arr = arr.astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
