git branch branch2
touch file4
git add file4
git commit -m"q11"
echo "123">file4
git stash
git checkout main
