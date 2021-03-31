#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

using namespace cv;

struct similarity_record {
	size_t image_a_idx;
	size_t image_b_idx;
	float avg_dist;
	float median_dist;
	size_t matches_count;
};

std::vector<DMatch> calculate_matches(const Ptr<ORB>& orb, const Ptr<BFMatcher>& bf, const Mat& a, const Mat& b, bool is_show = false) {
	std::vector<KeyPoint> keypointsA;
	Mat descriptorsA;
	orb->detectAndCompute(a, noArray(), keypointsA, descriptorsA);

	std::vector<KeyPoint> keypointsB;
	Mat descriptorsB;
	orb->detectAndCompute(b, noArray(), keypointsB, descriptorsB);

	std::vector<DMatch> matches;
	bf->match(descriptorsA, descriptorsB, matches);

	std::sort(matches.begin(), matches.end(), [](auto& l, auto& h){ return l.distance < h.distance; });

	if (is_show) {
		Mat matches_image;
		drawMatches(a, keypointsA, b, keypointsB, matches, matches_image);

		Mat matches_image_scaled;
		auto scaleX = matches_image.cols > 900 ? 900.f / static_cast<float>(matches_image.cols) : 1.f;
		auto scaleY = matches_image.rows > 1600 ? 1600.f / static_cast<float>(matches_image.rows) : 1.f;
		auto scale = std::min(scaleX, scaleY);

		resize(matches_image, matches_image_scaled, Size(), scale, scale);

		static int window_count = 0;
		imshow(std::to_string(window_count++), matches_image_scaled);
	}

	return matches;
}

float avg_distance(const std::vector<DMatch>& matches) {
	return std::accumulate(matches.begin(), matches.end(), 0.f, [](auto& sum, auto& match){
		return sum + match.distance;
	}) / matches.size();
}

float median_distance(const std::vector<DMatch>& matches) {
	return matches.empty() ? 0.f : matches[matches.size() / 2].distance;
}

std::vector<similarity_record> estimate_similarity(const std::vector<Mat>& images) {
	auto orb = ORB::create();
	auto bf = BFMatcher::create(NORM_HAMMING, true);

	std::vector<similarity_record> similarities;
	for (size_t i = 0; i < images.size(); i++) {
		for (size_t j = 0; j < images.size(); j++) {
			if (i == j) continue;

			auto matches = calculate_matches(orb, bf, images[i], images[j], true);

			similarities.push_back(
				similarity_record { i, j, avg_distance(matches), median_distance(matches), matches.size() }
			);
		}
	}

	return similarities;
}

int main() {
	float threshold;
	std::cin >> threshold;

	std::vector<std::string> image_paths;
	while (std::cin >> image_paths.emplace_back());
	image_paths.pop_back();

	if (image_paths.size() < 2) {
		std::cout << "Nothing to compare" << std::endl;
		return 0;
	}

	std::vector<Mat> images;
	for (const auto& image_path: image_paths) {
		auto image = imread(image_path, IMREAD_COLOR);
		if (image.empty()) {
			std::cout << "Invalid image path '" << image_path << "'" << std::endl;
			return 0;
		}
		images.push_back(image);
	}

	auto similarities = estimate_similarity(images);
	for (auto& s: similarities) {
		std::cout << image_paths[s.image_a_idx] << ",\t"
				  << image_paths[s.image_b_idx] << ",\t"
				  << std::fixed << std::setprecision(2)
				  << s.avg_dist << ",\t"
				  << s.median_dist << ",\t"
				  << s.matches_count << std::endl;
	}

	waitKey(0);

	return 0;
}
