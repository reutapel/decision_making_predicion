import pandas as pd
import os
import re

base_directory = os.path.abspath(os.curdir)
experiment_directory = os.path.abspath(os.path.join(base_directory, os.pardir))
text_experiment_directory = os.path.join(experiment_directory, 'text_experiment')


def fix_review(review_text: str) -> str:
    """
    This function get a review and:
    1. Add point before capital letters
    2. Add ' and remove space when: " ve ", " re ", " s ", " t ", " d ", " m ", " o "
    :param review_text:
    :return: the fix review
    """
    # 1. Add point before capital letters
    fix_review_text = re.sub(r" ([A-Z])",  r". \1", review_text)
    fix_review_text = re.sub(r"(Positive:. )", r"Positive: ", fix_review_text)
    fix_review_text = re.sub(r"(Negative:. )", r"Negative: ", fix_review_text)

    # 2. Add ' and remove space when: " ve ", " re ", " s ", " t ", " d ", " m ", " o "
    fix_review_text = re.sub(r" (t|s|o|ve|re|d|m|ll) ", r"'\1 ", fix_review_text)

    return fix_review_text


def fix_reviews(reviews_df: pd.DataFrame=None):
    if reviews_df is None:
        reviews_df = pd.read_excel(os.path.join(text_experiment_directory, 'fix_problems_text.xlsx'))
        reviews = reviews_df[['review1', 'review2', 'review3', 'review4', 'review5', 'review6', 'review7', 'review8',
                              'review9', 'review10']]
    else:
        reviews = reviews_df[['full_review']]
    for column_index, column in reviews.iteritems():
        column_name = column.name
        for row_index in column.index:
            fix_review_text = fix_review(column[row_index])
            reviews_df.loc[row_index, column_name] = fix_review_text

    reviews_df.to_excel(os.path.join(text_experiment_directory, 'reviews_fix_text.xlsx'))


def choose_reviews(number_reviews_per_hotel: int=9, min_char_in_review: int=100, max_char_in_review: int=800)\
        -> pd.DataFrame:
    all_reviews_df = pd.read_excel(os.path.join(text_experiment_directory, 'Hotel_Reviews.xlsx'),
                                   sheetname='Hotel_Reviews')
    all_reviews_df_length =\
        all_reviews_df.loc[all_reviews_df.review_length.between(min_char_in_review, max_char_in_review)]
    reviews_per_hotel = all_reviews_df_length.groupby(by='Hotel_Name').agg({'review_id': 'count'})
    reviews_per_hotel = reviews_per_hotel.loc[reviews_per_hotel.review_id >= number_reviews_per_hotel]
    sampled_reviews = pd.DataFrame(columns=all_reviews_df_length.columns)
    for hotel in reviews_per_hotel.index:
        hotel_reviews = all_reviews_df_length.loc[all_reviews_df_length.Hotel_Name == hotel]
        # print(f'hotel name: {hotel}, number of reviews: {hotel_reviews.shape}')
        hotel_reviews_sampled = hotel_reviews.sample(n=number_reviews_per_hotel)
        sampled_reviews = sampled_reviews.append(hotel_reviews_sampled, ignore_index=True)

    sampled_reviews.to_csv(os.path.join(text_experiment_directory, 'sampled_hotel_reviews.csv'))

    return sampled_reviews


def choose_organize_reviews(number_reviews_per_hotel):
    all_reviews_df = pd.read_excel(os.path.join(text_experiment_directory, 'reviews_fix_text.xlsx'),
                                   sheetname='final')
    reviews_per_hotel = all_reviews_df.groupby(by='Hotel_Name').agg({'review_id': 'count'})
    sampled_reviews = pd.DataFrame(columns=all_reviews_df.columns)
    columns_list = list()
    for i in range(number_reviews_per_hotel):
        columns_list.append(f'score_{i}')
        columns_list.append(f'review_{i}')
    sampled_organize_reviews = pd.DataFrame()

    for hotel in reviews_per_hotel.index:
        hotel_reviews = all_reviews_df.loc[all_reviews_df.Hotel_Name == hotel]
        # print(f'hotel name: {hotel}, number of reviews: {hotel_reviews.shape}')
        hotel_reviews_sampled = hotel_reviews.sample(n=number_reviews_per_hotel)
        hotel_reviews_sampled = hotel_reviews_sampled.sort_values(by=['Reviewer_Score'])
        sampled_reviews = sampled_reviews.append(hotel_reviews_sampled, ignore_index=True)
        reviews_list = list()
        for index, row in hotel_reviews_sampled.iterrows():
            reviews_list.append(row['Reviewer_Score'])
            reviews_list.append(row['full_review'])
        reviews_list = pd.Series(reviews_list)
        sampled_organize_reviews = sampled_organize_reviews.append(reviews_list, ignore_index=True)

    sampled_organize_reviews.columns = columns_list
    sampled_organize_reviews.to_csv(os.path.join(text_experiment_directory, 'reviews.csv'))
    sampled_reviews.to_excel(os.path.join(text_experiment_directory, 'seven_reviews.xlsx'))


def main():
    # sampled_reviews = choose_reviews(number_reviews_per_hotel=9, min_char_in_review=100, max_char_in_review=800)
    # fix_reviews(sampled_reviews)
    choose_organize_reviews(7)


if __name__ == '__main__':
    main()
