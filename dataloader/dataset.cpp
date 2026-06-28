#include "dataset.h"
#include <cstring>

void Dataset::get_images_batch(const std::vector<int>& indices,
                               float* dst) const {
    int img_sz = img_size();
    for (size_t i = 0; i < indices.size(); ++i) {
        get_image(indices[i], dst + i * img_sz);
    }
}
