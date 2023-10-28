# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)

import numpy as np
from PIL import Image

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    WithMetadata,
    WithWorkflow,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


@invocation("average_images", title="Average Images", tags=["image"], version="1.0.0")
class AverageImagesInvocation(BaseInvocation, WithMetadata, WithWorkflow):
    """Average images"""

    images: list[ImageField] = InputField(description="The collection of images to average")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        if len(self.images) == 0:
            raise ValueError("No input images specified")

        arr = None

        for this_image in self.images:
            image = context.services.images.get_pil_image(this_image.image_name)
            image = image.convert("RGB")
            if arr is None:
                w, h = image.size
                arr = np.zeros((h, w, 3), np.uint64)

            imarr = np.array(image, dtype=np.uint64)
            arr += imarr

        arr = arr / len(self.images)
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
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
