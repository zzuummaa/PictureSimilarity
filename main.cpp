#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <chrono>

using namespace cv;

struct similarity_record {
	size_t image_a_idx;
	size_t image_b_idx;
	float avg_dist;
	float median_dist;
	size_t matches_count;
	float metrics;
};

struct match_result {
	std::vector<DMatch> good;
	std::vector<DMatch> all;
};

match_result calculate_matches(const Ptr<ORB>& orb, const Ptr<BFMatcher>& bf, const Mat& a, const Mat& b, bool is_show = false) {
	std::vector<KeyPoint> keypointsA;
	Mat descriptorsA;
	orb->detectAndCompute(a, noArray(), keypointsA, descriptorsA);

	std::vector<KeyPoint> keypointsB;
	Mat descriptorsB;
	orb->detectAndCompute(b, noArray(), keypointsB, descriptorsB);

	std::vector<std::vector<DMatch> > matches;
	bf->knnMatch(descriptorsA, descriptorsB, matches, 2);

	float lowe_ratio = 0.89f;

	match_result result;
	for (const auto& match: matches) {
		if (match.size() != 2) {
			std::cout << "Invalid knnMatch output: match.size() != 2" << std::endl;
			return match_result();
		}
		if (match[0].distance < match[1].distance * lowe_ratio) result.good.push_back(match[0]);
		result.all.push_back(match[0]);
	}

	std::sort(result.good.begin(), result.good.end(), [](auto& l, auto& h){ return l.distance < h.distance; });
	std::sort(result.all.begin(), result.all.end(), [](auto& l, auto& h){ return l.distance < h.distance; });

	if (is_show) {
		Mat matches_image;
		drawMatches(a, keypointsA, b, keypointsB, result.good, matches_image);

		Mat matches_image_scaled;
		auto scaleX = matches_image.cols > 900 ? 900.f / static_cast<float>(matches_image.cols) : 1.f;
		auto scaleY = matches_image.rows > 1600 ? 1600.f / static_cast<float>(matches_image.rows) : 1.f;
		auto scale = std::min(scaleX, scaleY);

		resize(matches_image, matches_image_scaled, Size(), scale, scale);

		static int window_count = 0;
		imshow(std::to_string(window_count++), matches_image_scaled);
	}

	return result;
}

float avg_distance(const std::vector<DMatch>& matches) {
	if (matches.empty()) return 0.f;
	return std::accumulate(matches.begin(), matches.end(), 0.f, [](auto& sum, auto& match){
		return sum + match.distance;
	}) / matches.size();
}

float median_distance(const std::vector<DMatch>& matches) {
	return matches.empty() ? 0.f : matches[matches.size() / 2].distance;
}

float similarity_metrics(const match_result& result) {
	return static_cast<float>(result.good.size()) / result.all.size();
}

std::vector<similarity_record> estimate_similarity(const std::vector<Mat>& images) {
	auto orb = ORB::create();
	auto bf = BFMatcher::create();

	std::vector<similarity_record> similarities;
	for (size_t i = 0; i < images.size(); i++) {
		for (size_t j = i + 1; j < images.size(); j++) {
			auto matches = calculate_matches(orb, bf, images[i], images[j], false);

			similarities.push_back(
				similarity_record {
						i,
						j,
						avg_distance(matches.all),
						median_distance(matches.all),
						matches.all.size(),
						similarity_metrics(matches)
				}
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
		if (s.metrics * 100 < threshold) continue;

		std::cout << image_paths[s.image_a_idx] << ",\t"
				  << image_paths[s.image_b_idx] << ",\t"
//				  << std::fixed << std::setprecision(2)
//				  << s.avg_dist << ",\t"
//				  << s.median_dist << ",\t"
//				  << s.matches_count << ",\t"
				  << static_cast<int>(s.metrics * 100) << std::endl;
	}

	return 0;
}
